
# ===================================================================================
#                            Mie Calculation
# ===================================================================================

# calculate mie theory
if mieroutine == 'miepython':
    # Use miepython
    import miepython
    m_eff = complex(eff[0], -eff[1])
    x_fac = 2 * np.pi * cloud_radius / wavelength
    qe_temp, qs_temp, _, _ = miepython.mie(m_eff, x_fac)
        
        
# ===================================================================================
#                            Effective Medium Calculation
# ===================================================================================

# loop over all height points
for wav in range(len_w):
    for hig in range(len_h):

        # bruggemann effective medium theory
        # sum(vol_frac*(e_i - e_eff)/(e_i + 2*e_eff))=0
        brug_failed = False
        if emt_routine in ['Bruggeman', 'Bruggeman-LLL']:
            print('[INFO] Bruggeman progress: '
                  + str(round((hig + wav*len_h)/len_h/len_w*100, 1)) +
                  ' %', end='\r')

            # initial guess using linear approximation
            # comment for future me: the divide by 10 is necessary for heterogenous
            # cloud particles with a high ref component like iron. This devid by
            # 10 helps to find the actual minima and not just a boundary condition.
            m_eff_0 = np.zeros((2,))
            m_eff_0[0] = sum(work[:, wav, hig, 0] * work[:, wav, hig, 1])/10
            m_eff_0[1] = sum(work[:, wav, hig, 0] * work[:, wav, hig, 2])/10

            # define constraints
            def con_min_n(eff, work):
                return eff[0] - np.min(work[:, 1])

            def con_max_n(eff, work):
                return np.max(work[:, 1]) - eff[0]

            def con_min_k(eff, work):
                return eff[1] - np.min(work[:, 2])

            def con_max_k(eff, work):
                return np.max(work[:, 2]) - eff[1]

            con = [{'type': 'ineq', 'fun': con_min_n, 'args': [work[:, wav, hig, :]]},
                   {'type': 'ineq', 'fun': con_max_n, 'args': [work[:, wav, hig, :]]},
                   {'type': 'ineq', 'fun': con_min_k, 'args': [work[:, wav, hig, :]]},
                   {'type': 'ineq', 'fun': con_max_k, 'args': [work[:, wav, hig, :]]},
                   ]

            # calculate effective medium theory with Bruggemann minimization
            resul = minimize(_func, m_eff_0, args=work[:, wav, hig, :],
                             constraints=con, method='SLSQP')#, tol=1e-12)
            m_eff = resul.x

            # save success variable since we will manipulate it later
            success = resul.success

            # check if real part is within range of all material composition
            tollarance = 1e-5
            if m_eff[0] < np.min(work[:, wav, hig, 1]) - tollarance:
                print('\n[WARN] n_eff is below all n_i.')
                success = False
            if m_eff[0] > np.max(work[:, wav, hig, 1] + tollarance):
                print('\n[WARN] n_eff is above all n_i.')
                success = False
            if m_eff[1] < np.min(work[:, wav, hig, 2] - tollarance):
                print('\n[WARN] k_eff is below all k_i.')
                success = False
            if m_eff[1] > np.max(work[:, wav, hig, 2] + tollarance):
                print('\n[WARN] k_eff is above all k_i.')
                success = False

            # here, negative values can still occure due to tollarances. Prevent these.
            if m_eff[0] < 0:
                m_eff[0] = 0
            if m_eff[1] < 0:
                m_eff[1] = 0

            # if succeeded save value, otherwise go to special case handling
            if success:
                eff[wav, hig] = m_eff
            else:
                # if only bruggemann is allowed, abbort here
                if emt_routine == 'Bruggeman':
                    raise ValueError('[ERROR] Failieur of Brugeman minization at '
                                     + str(wavelength[wav]*1e4) + ' micron.')
                elif emt_routine == 'Bruggeman-LLL':
                    print('\n[WARN] Failieur of Brugeman minization at '
                          + str(wavelength[wav]*1e4) + ' micron. Falling back to LLL.\n')
                    brug_failed = True


        # LLL method
        # n^2 = e = (sum(vol_frac*e_cur**(1/3)))**3
        if emt_routine == 'LLL' or (emt_routine == 'Bruggeman-LLL' and brug_failed):
            e_eff = complex(0, 0)
            for k, key in enumerate(volume_fraction):
                # convert to dielectric constant
                e_cur = complex(work[k, wav, hig, 1], work[k, wav, hig, 2])**2
                # calculate effecitve dielectric constant
                e_eff += work[k, wav, hig, 0] * e_cur**(1./3.)
            e_eff = e_eff ** 3
            # convert back to refractiv index
            m_eff = e_eff ** (1./2.)

            # save results
            eff[wav, hig, 0] = m_eff.real
            eff[wav, hig, 1] = m_eff.imag
            
 
# ===================================================================================
#                            Data read in
# ===================================================================================           

def _cloud_nk_data(claus, species, wavelength):
    """
    Get the real (n) and imaginary (k) refractory index of a given species.

    Parameters
    ----------
    species: str
        Name of the cloud particle species. Currently supported:
        TiO2[s], Mg2SiO4[s], SiO[s], SiO2[s], Fe[s], Al2O3[s], CaTiO3[s], FeO[s],
        FeS[s], Fe2O3[s], MgO[s], MgSiO3[s], CaSiO3[s], Fe2SiO4[s], C[s], KCl[s]
    wavelength: Union[np.ndarray, float]
        The wavelength at which the refractory index should be given [cm].

    Returns
    -------
    ref_index: np.ndarray[n, 2]
        The refractory index with the first index being the wavelenght and the second being n, k.
    """

    # gather all data files
    files = glob(os.path.dirname(__file__) + '/../nk_data/*.dat')

    # convert wavelengths to micron
    wave_temp = wavelength * 1e4

    # prepare output
    if not isinstance(wave_temp, np.ndarray):
        wave_temp = np.asarray([wave_temp])
    ref_index = np.zeros((len(wave_temp), 2))

    # find species in files
    for fil in files:
        if species in fil:

            # get data
            content = open(fil, 'r').readlines()
            header = content[0].split()[1]
            data = content[5:]

            # loop over all wavelengths:
            for wav, wave in enumerate(wave_temp):
                # if desired wavelength is smaller than data, use the smallest
                # wavelength data available.
                if wave < float(data[0].split()[0]):
                    ref_index[wav, 0] = float(data[0].split()[1])
                    ref_index[wav, 1] = float(data[0].split()[2])
                    continue

                # if wavelength is within range, log-log interpolation
                for dnr, _ in enumerate(data):
                    cur_wave = float(data[dnr].split()[0])  # current wavelength
                    if wave < cur_wave:
                        nlo = float(data[dnr - 1].split()[1])  # lower n value
                        nhi = float(data[dnr].split()[1])  # higher n value
                        klo = float(data[dnr - 1].split()[2])  # lower k value
                        khi = float(data[dnr].split()[2])  # higher k value
                        prev_wave = float(data[dnr - 1].split()[0])  # previous wavelength

                        # calculate interpolation
                        fac = np.log(wave / prev_wave) / np.log(cur_wave / prev_wave)
                        ref_index[wav, 0] = np.exp(np.log(nlo) + fac * np.log(nhi / nlo))
                        if klo <= 0 or khi <= 0:
                            ref_index[wav, 1] = 0
                        else:
                            ref_index[wav, 1] = np.exp(np.log(klo) + fac * np.log(khi / klo))

                        break

                else:
                    # if wavelength is outside of range, extrapolate
                    if header == '.True.':
                        # conducting interpolation, log-log interpolation
                        # find 70% wavelength
                        cur_wave = float(data[0].split()[0])  # current wavelength
                        max_wave = float(data[-1].split()[0])  # maximum wavelength
                        c_nr = 0
                        for dnr, _ in enumerate(data):
                            cur_wave = float(data[dnr].split()[0])  # current wavelength
                            if cur_wave < 0.7 * max_wave:
                                break
                            c_nr += 1
                        nlo = float(data[c_nr].split()[1])  # lower n value
                        nhi = float(data[-1].split()[1])  # higher n value
                        klo = float(data[c_nr].split()[2])  # lower k value
                        khi = float(data[-1].split()[2])  # higher k value

                        # calculate interpolation
                        fac = np.log(wave / max_wave) / np.log(cur_wave / max_wave)
                        ref_index[wav, 0] = np.exp(np.log(nhi) + fac * np.log(nlo / nhi))
                        if klo <= 0 or khi <= 0:
                            ref_index[wav, 1] = 0
                        else:
                            ref_index[wav, 1] = np.exp(np.log(khi) + fac * np.log(klo / khi))
                    else:
                        # non-conducting interpolation, linear decreasing k, constant n
                        ref_index[wav, 0] = float(data[-1].split()[1])
                        ref_index[wav, 1] = float(data[-1].split()[2]) * \
                                            float(data[-1].split()[0]) / wave

            # stop searching the files
            break
    else:
        # if no data was found, throw error
        print(
            "[WARN]",
            'The following cloud particle bulk species is currently ' +
            'not supported or misspelled: ' + species + '. From here ' +
            'on it is ignored in further calculations.'
        )
        claus.no_data_cloud_species.append(species)
        ref_index[:, 0] = 1
        ref_index[:, 1] = 0

    # return refractive indices
    return ref_index
