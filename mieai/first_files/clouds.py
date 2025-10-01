"""
==============================================================
              Cloud calculation functionalities
==============================================================
 This file contains multiple functionalities to deal with
 clouds.
==============================================================
"""
# pylint: disable=W0212,R0902,R0913,R0912,C0415,R1702
import os
from glob import glob
import numpy as np
from scipy.optimize import minimize


def cloud_opacities(claus, wavelengths, cloud_radius, cloud_abundances, cloud_particle_density,
                    volume_fraction):
    """
    Calculate cloud particle opacities using Mie-Theory. This function assumes spherical
    particles. The output are the absorption efficiency, scattering efficiency, and
    cross-section. One can calculate the Kappa values by multiplying the efficiencies
    with the cross-section.

    Parameters
    ----------
    wavelengths: np.ndarray[w]
        wavelengths to evaluate the opacities at [cm].
    cloud_radius: np.ndarray[h]
        Mean cloud particle radius [cm].
    cloud_abundances: np.ndarray[h]
        Mass fraction of clouds compared to the gas.
    cloud_particle_density: np.ndarray[h]
        Bulk density of cloud particles [g/cm^3]
        (for homogenous cloud particles, this is the same as the density of
        the cloud particle material)
    volume_fraction: dict[Species: np.ndarray[h]]
        Volume fractions of cloud species. All species included must be
        supported by the cloud_nk_data function.

    Returns
    -------
    Qabs : np.ndarray[h, w]
        Absorption coefficients
    Qsca : np.ndarray[h, w]
        Scattering coefficients
    c_sec : np.ndarray[h]
        cross sections [cm^2/g]
    """

    # check input
    if not len(cloud_radius) == len(cloud_abundances) == len(cloud_particle_density):
        raise ValueError('All cloud particle properties need to be '
                         'defined on the same pressure gird and have '
                         'the same number of entries. Given:'
                         '\n  len(cloud_radius) = ' + str(len(cloud_radius)) +
                         '\n  len(cloud_abundances) = ' + str(len(cloud_abundances)) +
                         '\n  len(cloud_particle_density) = ' + str(len(cloud_particle_density)))

    # low number handling
    cloud_particle_density[cloud_particle_density < 0] = 0
    cloud_radius[cloud_radius < 0] = 0

    # get array lengths
    len_h = len(cloud_radius)
    len_w = len(wavelengths)

    # prepare output arrays
    qabs = np.zeros((len_h, len_w))
    qsca = np.zeros((len_h, len_w))

    # calculate effective medium theory for each height and each wavelength
    eff = eff_func(claus, volume_fraction, wavelengths, claus.emtroutine)

    # cloud particle cross section
    c_sec = np.zeros((len_h,))

    # loop over all inputs:
    for wav in range(len_w):
        for hig in range(len_h):

            # handle zero sized cloud particles
            if cloud_radius[hig] <= 0:
                qabs[hig, wav] = 0
                qsca[hig, wav] = 0
                c_sec[hig] = 0
                continue

            # calculate cloud particle cross section
            # not to self because this has been derived one too many times:
            # c_sec = [cloud particle cross section per gas mass]
            #       = [csec. per cloud part.] * [number density of cpart.] / [gas density]
            #       =  pi r^2                 *  n_d                       /  rho_g
            # n_d   = [number density of cpart.]
            #       = [mass of cpart. per volume] / [mass of 1 cpart.]
            #       =  rho_dg                     /   m_d
            # m_d   = [mass of 1 cpart.]
            #       = [volume of 1 cpart.] * [density of cpart.]
            #       =  4/3 pi r^3          *  rho_d
            # c_sec = pi r^2 * rho_dg / (4/3 pi r^3 * rho_d * rho_g)
            #       = (rho_dg/rho_g) * 3 / (4 * r * rho_d)
            c_sec[hig] = (cloud_abundances[hig] * 3 / 4
                          / (cloud_radius[hig] * cloud_particle_density[hig]))

            # BAS assumes different material rather than a single mixed
            # material. Therefore, mie theory has to be calcualted per material.
            if claus.emtroutine == 'BAS':
                qe_temp = 0
                qs_temp = 0
                for k, key in enumerate(volume_fraction):
                    # volume of a sphere with the volume fraction material
                    cl_radius_i = cloud_radius[hig] * (volume_fraction[key][hig])**(1/3)
                    # from here just normal mie for each component
                    qe_tete, qs_tete = mie_func(eff[k, wav, hig, :], wavelengths[wav],
                                                cl_radius_i, claus.mieroutine,
                                                claus.dhs_max_frac)

                    # here we add a correction factor for the actual crossesction of
                    # each particle material compared to the total cross section of
                    # the particle.
                    qe_temp += qe_tete * cl_radius_i**2 / cloud_radius[hig]**2
                    qs_temp += qs_tete * cl_radius_i**2 / cloud_radius[hig]**2

            # System of Sphere approximation assumes that all cloud particle are
            # homogenous and their number density is proportional to the volume fraction.
            elif claus.emtroutine == 'SSA':
                qe_temp = 0
                qs_temp = 0
                for k, key in enumerate(volume_fraction):
                    # calculate the mie opacities for each particle seperatly
                    qe_tete, qs_tete = mie_func(eff[k, wav, hig, :], wavelengths[wav],
                                                cloud_radius[hig], claus.mieroutine,
                                                claus.dhs_max_frac)

                    # here we adjust the cross section by the volume fraction
                    qe_temp += qe_tete * volume_fraction[key][hig]
                    qs_temp += qs_tete * volume_fraction[key][hig]

            # Calculation for a Core-Shell model
            elif claus.emtroutine == 'Core-Shell':
                # fortran function call
                from . import dmilay

                # default values
                qe_temp = 0
                qs_temp = 0

                # core index
                coi = claus.core_shell[1]

                # the core needs to be distributed to each shell material
                core_volume_adjustment = len(volume_fraction.keys()) - 1
                if core_volume_adjustment <= 0:
                    raise ValueError("Core-Shell calculation requires at least " +
                                     "two cloud particle materials.")

                # calculation of core radius
                # from: vol_frac / nr_shell_species 4/3 pi r_cloud^3 = 4/3 pi r_in^3
                rcore = cloud_radius[hig] * (eff[coi, wav, hig, 0])**(1/3)

                for k, _ in enumerate(volume_fraction):
                    # skip core material
                    if k == claus.core_shell[1]:
                        continue

                    # calculation of shell radius
                    # from: V = 4/3 pi (r_out^3 - r_in^3) = vol_frac 4/3 pi r_cloud^3
                    rshell = (cloud_radius[hig] *
                              (eff[k, wav, hig, 0]*core_volume_adjustment +
                               eff[coi, wav, hig, 0])**(1/3))

                    # if there is no core size or shell size, there is no cloud particle
                    if rcore <= 0 and rshell <= 0:
                        qe_tete = 0
                        qs_tete = 0
                        continue

                    # if there is only a core or a shell size, this is a homogenous particle
                    elif rcore <= 0:
                        qe_tete, qs_tete = mie_func([eff[k, wav, hig, 1], eff[1, wav, hig, 2]],
                                                    wavelengths[wav], rshell, claus.mieroutine,
                                                    claus.dhs_max_frac)
                    elif rshell <= 0:
                        qe_tete, qs_tete = mie_func([eff[coi, wav, hig, 1], eff[0, wav, hig, 2]],
                                                    wavelengths[wav], rcore, claus.mieroutine,
                                                    claus.dhs_max_frac)

                    # if there is a core and shell size, this is a mixed particle
                    else:
                        # wavenumber
                        wvno = 2 * np.pi / wavelengths[wav]

                        # refractive indices of core and shell
                        ref_core = complex(eff[coi, wav, hig, 1], -eff[coi, wav, hig, 2])
                        ref_shell = complex(eff[k, wav, hig, 1], -eff[k, wav, hig, 2])

                        qe_tete, qs_tete = dmilay.dmilay(rcore, rshell, wvno, ref_shell, ref_core)

                    # adjust the number denisty so that each shell material has the same number
                    qe_tete /= core_volume_adjustment
                    qs_tete /= core_volume_adjustment

                    # add and adjust for smaller cross section size
                    qe_temp += qe_tete / cloud_radius[hig]**2 * rshell**2
                    qs_temp += qs_tete / cloud_radius[hig]**2 * rshell**2

                    # check for nans
                    qe_temp = np.nan_to_num(qe_temp)
                    qs_temp = np.nan_to_num(qs_temp)

            # if cloud particles are mixed, calculat mie with emt theory
            else:
                print('[INFO] Mie progress: '
                      + str(round((hig + wav*len_h)/len_h/len_w*100, 1)) +
                      ' %', end='\r')
                # mie calculation
                qe_temp, qs_temp = mie_func(eff[wav, hig, :], wavelengths[wav],
                                            cloud_radius[hig], claus.mieroutine,
                                            claus.dhs_max_frac)
            qabs[hig, wav] = qe_temp - qs_temp
            qsca[hig, wav] = qs_temp

    # all done
    print(' '*40, end='\r')
    print('[INFO] Mie done')
    return qabs, qsca, c_sec


def mie_func(eff, wavelength, cloud_radius, mieroutine, dhs_fmax=None):
    """
    Mie calculation using different functions.

    :param eff: effective refractive np.ndarray(2)
    :param wavelength: wavelength to be evaluated [cm]
    :param cloud_radius: cloud particle radius [cm]
    :param mieroutine: string detailing the mieroutine
    :param dhs_fmax: maximum fraction for DHS calculation
    :return: qabs, qext
    """
    # calculate mie theory
    if mieroutine == 'miepython':
        # Use miepython
        import miepython
        m_eff = complex(eff[0], -eff[1])
        x_fac = 2 * np.pi * cloud_radius / wavelength
        qe_temp, qs_temp, _, _ = miepython.mie(m_eff, x_fac)

    elif mieroutine == 'PyMieScatt':
        raise ValueError('PyMieScatt support is currently disabled because the library has '
                         'not been updated to the newest scipy version')
        # # Use PyMieScatt
        # import PyMieScatt as ps
        # m_eff = complex(eff[0], eff[1])
        # qe_temp, qs_temp, _, _, _, _, _ \
        #     = ps.MieQ(m_eff, wavelength*1e7, 2*cloud_radius*1e7)

    elif mieroutine == 'Miex':
        # use Miex
        from . import miex
        m_eff = complex(eff[0], eff[1])
        x_fac = 2 * np.pi * cloud_radius / wavelength
        qe_temp, qs_temp, _, _, _, _, _, _ \
            = miex.shexqnn2(m_eff, x_fac, False, 1)
    elif mieroutine == 'DHS':
        # check if DHS maxfrac is given
        if dhs_fmax is None:
            raise ValueError('DHS calculation requries dhs_fmax parameter')
        # use Distribution of hollow spheres
        from . import qdhs
        qe_temp, qs_temp = qdhs.qdhs(eff[0], eff[1],
                                       wavelength*1e4,  # wavelength in microns
                                       cloud_radius*1e4,  # radius in microns
                                       dhs_fmax)
        if cloud_radius <= 0:
            qe_temp = 0
            qs_temp = 0
        else:
            # DHS module actually returns cross sections because it considers multiple
            # sized particles. Here we correct for the avarage particle size to be
            # consistent with later processing.
            csec = (cloud_radius*1e4)**2 * np.pi
            qe_temp /= csec
            qs_temp /= csec
    else:
        raise ValueError('Mie routine selected not recognised. This error should ' +
                         'only occure if you manipulated claus.mieroutine outside ' +
                         'of the legal functions.')

    return qe_temp, qs_temp


def eff_func(claus, volume_fraction, wavelength, emt_routine='LLL'):
    """
    Get the effective real and imaginary refractory index of a mixed
    cloud particle grain. As default, the LLL method is used.

    Parameters
    ----------
    volume_fraction: dict[Species: np.ndarray[h]]
        Volume fractions of cloud species. All species included must be
        supported by the cloud_nk_data function.
    wavelength: np.ndarray[w]
        The wavelength at which the refractory index should be given [cm].
    emt_routine : str
        Type of effective medium theory used.

    Returns
    -------
    ref_index: np.ndarray[w, h, 2]
        The refractory index (real, imaginary) with the first index
        being the wavelength, second index being the height point.
    """

    # ===================================================================================
    #                           General opacity data read in
    # ===================================================================================

    # get number of height points
    len_h = len(volume_fraction[list(volume_fraction.keys())[0]])
    # get number of wavelengths
    len_w = len(wavelength)
    # get number of bulk material species
    len_b = len(volume_fraction.keys())

    # Maxwell-Garnett warning flag to prevent to many warnings
    mg_warning_flag = True

    # prepare output
    eff = np.zeros((len_w, len_h, 2))

    # get all ref indexes:
    # [material, wavelength, height, [volume fraction, n, k]]
    work = np.zeros((len_b, len_w, len_h, 3))
    for k, key in enumerate(volume_fraction):

        # check if a species should be forced to vacuum
        if key in claus.force_vacuum:
            print('[INFO] The species ' + key + ' was manualy set to vaccum.')
            temp = np.ones((len(wavelength), 2))
            temp[:, 1] = 0
        else:
            temp = _cloud_nk_data(claus, key, wavelength)

        # assign vlaues
        for hig in range(len_h):
            work[k, :, hig, 1:] = temp

        # for core shell, remember index
        if emt_routine == "Core-Shell":
            if key == claus.core_shell[0]:
                claus.core_shell[1] = k

    # get all volume fractions
    for wav in range(len_w):
        for hig in range(len_h):
            for k, key in enumerate(volume_fraction):
                # if no cloud opacity data was found for the given species,
                # set the volume fraction to 0
                if key in claus.no_data_cloud_species:
                    work[k, wav, hig, 0] = 0
                else:
                    work[k, wav, hig, 0] = volume_fraction[key][hig]

    # ===================================================================================
    #                                Special output cases
    # ===================================================================================

    # BAS and SSA do not require actual emt calculation but rather
    # returingn all refractive indeces. This is done here
    if emt_routine in ['BAS', 'SSA']:
        eff = work[:, :, :, 1:]
        return eff

    # Core-Shell only requires two materials which are returned here
    if emt_routine == "Core-Shell":
        # check if core and shell materials were loaded
        if claus.core_shell[1] == -1:
            raise ValueError("Core species (" + claus.core_shell[0] + ") was not found in data.")

        # warining if there are more than two materials
        if len(volume_fraction) > 2:
            print('[WARN] Core-Shell was selected but the cloud particles are made ' +
                  'from more than two materials. Multiple two component materials ' +
                  'will be considered')

        return work

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

            if emt_routine == 'Maxwell-Garnett':
                # Maxwell-Garnett is only valid for small inclusion. Thus, first determine
                # dominant material composisiton.
                dom_i = work[:, wav, hig, 0].argmax()

                # check if species is dominant (here defined as vol_frac > 1-1e-4 (Belyaev &
                # Tyurnev 2018, Journal of Experimental and Theoretical Physics, 127, 608)).
                if work[dom_i, wav, hig, 0] < 1-1e-4 and mg_warning_flag:
                    print('[WARN] Maxwell-Garnett emt was selected but largest volum ' +
                          'fraction is outside of the validity.')
                    mg_warning_flag = False

                # get dominant material refractive index:
                e_dom = complex(work[dom_i, wav, hig, 1], work[dom_i, wav, hig, 2])**2

                # calculate rhs of Maxwell-Garnett equation
                gamma = complex(0, 0)
                for k, key in enumerate(volume_fraction):
                    # skip dominant material
                    if k == dom_i:
                        continue
                    # current refractiv index
                    e_cur = complex(work[k, wav, hig, 1], work[k, wav, hig, 2])**2
                    # calculate contribution from current material
                    gamma += work[k, wav, hig, 0] * (e_cur - e_dom)/(e_cur + 2*e_dom)

                # calculate effective medium
                e_eff = e_dom * (1 + 2*gamma)/(1- gamma)

                # convert back to refractive index
                m_eff = e_eff ** (1./2.)

                # save results
                eff[wav, hig, 0] = m_eff.real
                eff[wav, hig, 1] = m_eff.imag

            if emt_routine == 'Linear':
                # print a warning if Linear is used since this is only a test routine
                if mg_warning_flag:
                    print('[WARN] You selected Linear EMT. Consider using LLL instead.')
                    mg_warning_flag = False

                eff[wav, hig, 0] = sum(work[:, wav, hig, 0] * work[:, wav, hig, 1])
                eff[wav, hig, 1] = sum(work[:, wav, hig, 0] * work[:, wav, hig, 2])

    print(' '*40, end='\r')
    print('[INFO] EMT done')
    return eff


# define Bruggeman mixing rule
def _func(eff, work):
    """
    Bruggeman mixing function.

    Parameters
    ----------
    eff: np.ndarray[2]
        effective reffrective index. First index real part, second index
        imaginary part
    work: np.ndarray[s, 3]
        The volume mixing fraction, real refractive index and imaginary
        refractive index for each bulk material species s.

    Returns
    -------
    quality_of_fit : float
        A quality of fit parameter. Should be as close to zero as possible.
    """
    sol = complex(0, 0)

    for i, _ in enumerate(work):
        # get dielectic constants from refractive index
        e_i = complex(work[i, 1], work[i, 2])**2
        e_e = complex(eff[0], eff[1])**2

        # add to sumation
        sol += work[i, 0]*(e_i-e_e)/(e_i+2*e_e)

    return abs(sol)#sol.real**2 + sol.imag**2


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
    if not isinstance(wave_temp, ndarray):
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
