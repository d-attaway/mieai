import numpy as np

def mixing_theory(wavelength, ref_index, vmr):
    # prepare outputs
    mixed_ref_index = np.zeros(len(wavelength), dtype=complex)

    # loop over wavelength
    for wav, wave in enumerate(wavelength):
        # dielectric constant = (n + ik)^2
        ind_eff = (ref_index[:, wav, 0] + (1j * ref_index[:, wav, 1])) ** 2
        e_eff = (np.sum(vmr[wav] * (ind_eff ** (1 / 3)))) ** 3

        # find mixed refractive index, real(n) and imaginary(k)
        m_eff = np.sqrt(e_eff)
        mixed_ref_index[wav] = complex(m_eff.real, -m_eff.imag)

    return mixed_ref_index