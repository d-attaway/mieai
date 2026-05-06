import numpy as np
from scipy.optimize import minimize

def mixing_theory(wavelength, ref_index, vmr, theory='LLL'):
    """
    Calculate refractive index of mixed species.

    Parameters
    ----------
    wavelength: np.ndarray[N]
        Wavelenght grid
    ref_index: np.ndarray[M, N, 2]
        Refractive index at each wavlength and for each material
    vmr: np.ndarray[N, M]
        The volume mixing ratio of each of the M species
    theory : str, optional
        Mixing theory used, can either be 'LLL' (Default) or 'Burggeman'

    Returns
    -------
    mixed_ref_index : np.ndarray[N]
        Complex array with the refractive index of the mixed material
    """
    # prepare outputs
    mixed_ref_index = np.zeros(len(wavelength), dtype=complex)

    # loop over wavelength
    for wav, wave in enumerate(wavelength):
        if theory == 'LLL':
            # dielectric constant = (n + ik)^2
            ind_eff = (ref_index[:, wav, 0] + (1j * ref_index[:, wav, 1])) ** 2
            e_eff = (np.sum(vmr[wav] * (ind_eff ** (1 / 3)))) ** 3

            # find mixed refractive index, real(n) and imaginary(k)
            m_eff = np.sqrt(e_eff)
            mixed_ref_index[wav] = complex(m_eff.real, -m_eff.imag)

        elif theory == 'Bruggeman':
            # initial guess is 0.1 of linear mixing
            m_eff_0 = np.asarray([np.sum(vmr[wav] * ref_index[:, wav, 0]),
                                  np.sum(vmr[wav] * ref_index[:, wav, 1])])
            # define constraints
            def con_min_n(nk, n_min): return nk[0] - n_min
            def con_max_n(nk, n_max): return n_max - nk[0]
            def con_min_k(nk, k_min): return nk[1] - k_min
            def con_max_k(nk, k_max): return k_max - nk[1]
            con = [{'type': 'ineq', 'fun': con_min_n, 'args': [np.min(ref_index[:, wav, 0])]},
                   {'type': 'ineq', 'fun': con_max_n, 'args': [np.max(ref_index[:, wav, 0])]},
                   {'type': 'ineq', 'fun': con_min_k, 'args': [np.min(ref_index[:, wav, 1])]},
                   {'type': 'ineq', 'fun': con_max_k, 'args': [np.max(ref_index[:, wav, 1])]},
                   ]

            # calculate effective medium theory with Bruggemann minimization
            work = [vmr[wav], ref_index[:, wav, 0], ref_index[:, wav, 1]]
            resul = minimize(bruggeman_func, m_eff_0, args=[],
                             constraints=con, method='SLSQP')
            m_eff = resul.x

            # here, negative values can still occure due to tollarances. Prevent these.
            if m_eff[0] < 0: m_eff[0] = 0
            if m_eff[1] < 0: m_eff[1] = 0

            # check if it worked
            if not resul.success:
                raise ValueError('[ERROR] Failieur of Brugeman minization at '
                                 + str(wavelength[wav] * 1e4) + ' micron.')

            # save results
            mixed_ref_index[wav] = complex(m_eff[0], -m_eff[1])

        else:
            raise ValueError("Unknown mixing theory '" + theory + "'")

    return mixed_ref_index



# define Bruggeman mixing rule
def bruggeman_func(eff, work):
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

