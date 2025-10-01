import os
import numpy as np
from tensorflow import keras
from keras import layers
import glob
import miepython as mie


class Mieai:
    def __init__(self):
        # ==== Define ML model ==========================================================
        inputs = keras.Input(shape=(4,), name='inputs')
        # layers
        hidden1 = layers.Dense(100, activation='gelu', kernel_initializer='he_normal')(inputs)
        hidden2 = layers.Dense(100, activation='gelu', kernel_initializer='he_normal')(hidden1)
        # extinction branch
        ext_hidden1 = layers.Dense(100, activation='gelu', kernel_initializer='he_normal')(hidden2)
        ext_hidden2 = layers.Dense(100, activation='gelu', kernel_initializer='he_normal')(ext_hidden1)
        ext_hidden3 = layers.Dense(100, activation='gelu', kernel_initializer='he_normal')(ext_hidden2)
        ext_hidden4 = layers.Dense(100, activation='gelu', kernel_initializer='he_normal')(ext_hidden3)
        output1 = layers.Dense(1, activation='softplus', name='extinction')(ext_hidden4)
        # scattering branch
        sca_hidden1 = layers.Dense(100, activation='gelu', kernel_initializer='he_normal')(hidden2)
        sca_hidden2 = layers.Dense(100, activation='gelu', kernel_initializer='he_normal')(sca_hidden1)
        sca_hidden3 = layers.Dense(100, activation='gelu', kernel_initializer='he_normal')(sca_hidden2)
        sca_hidden4 = layers.Dense(100, activation='gelu', kernel_initializer='he_normal')(sca_hidden3)
        output2 = layers.Dense(1, activation='softplus', name='scattering')(sca_hidden4)
        # asymmetry branch
        asym_hidden1 = layers.Dense(100, activation='gelu', kernel_initializer='he_normal')(hidden2)
        asym_hidden2 = layers.Dense(100, activation='gelu', kernel_initializer='he_normal')(asym_hidden1)
        asym_hidden3 = layers.Dense(100, activation='gelu', kernel_initializer='he_normal')(asym_hidden2)
        asym_hidden4 = layers.Dense(100, activation='gelu', kernel_initializer='he_normal')(asym_hidden3)
        output3 = layers.Dense(1, name='asymmetry')(asym_hidden4)
        # make model
        self.model = keras.Model(inputs, outputs=[output1, output2, output3])

        # ==== Load species data from files ==============================================================
        self.files = glob.glob(os.path.dirname(__file__) + '/../first_files/nk_data/*.dat')
        self.species_list = ['TiO2[s]', 'Fe[s]', 'Mg2SiO4[s]']


    def ai_efficiencies(self, wavelength, particle_size, volume_mixing_ratios):
        """
        Calculate mie coefficients using a pre-trained neural network.

        Parameters
        ----------
        wavelength : np.ndarray or float
            Wavelength of the light [micron]
        particle_size : np.ndarray or float
            Size of the cloud particle [micron]
        volume_mixing_ratios : dict of np.ndarray or float
            Fraction of each cloud material given as float or array

        Return
        ------
        optical properties : np.ndarray
            extinction coefficient, scattering coefficient, and asymmetries parameter
        """

        # ==== Input checks =============================================================

        # check inputs are correct type
        if not isinstance(wavelength, np.ndarray) and not isinstance(wavelength, (float, int)):
            print('Wavelength must be of type np.ndarray or float')
        if not isinstance(particle_size, np.ndarray) and not isinstance(particle_size, (float, int)):
            print('Particle size must be of type np.ndarray or float')
        if not isinstance(volume_mixing_ratios, dict) and not isinstance(volume_mixing_ratios, (float, int)):
            print('Volume mixing ratio must be of type dict or float')

        # convert floats to arrays
        if isinstance(wavelength, (float, int)):
            wavelength = np.array([wavelength])
        if isinstance(particle_size, (float, int)):
            particle_size = np.array([particle_size])

        # make all possible combinations of wavelength & particle size
        final_wavelength = np.repeat(wavelength, len(particle_size))
        final_particle_size = np.tile(particle_size, len(wavelength))

        #adjust volume mixing ratio
        vmr = np.array(list(volume_mixing_ratios.values())).T

        if len(set(map(len, volume_mixing_ratios.values()))) != 1:
            print('Volume mixing ratios must have same shape')

        vmr_sum = np.sum(vmr, axis=1)
        if any(vmr_sum != 1):
            print('Volume mixing ratios do not add up to 1. The ratios have been renormalized.')
        idx = np.asarray(np.where(vmr_sum != 1))
        for i in idx[0]:
            vmr[i] = vmr[i] / sum(vmr[i])

        # make volume mixing ratios have the same dimensions as final wavelength & final particle size
        final_vmr = np.tile(vmr, (len(wavelength), 1))

        # ==== Prepare model ============================================================
        # ==== TODO: find a way to quickly search for the right model
        self.model.load_weights(os.path.dirname(__file__) + '/models/model47.weights.h5')

        # ==== define input array TODO: generalize this for all models
        inputs = np.stack((np.asarray(np.log10(final_wavelength)),
                           np.asarray(np.log10(final_particle_size)),
                           np.asarray(final_vmr[:,0]),
                           np.asarray(final_vmr[:,1])), axis=1)

        outputs = self.model.predict(inputs)
        qext = outputs[0][:, 0].reshape((len(wavelength), len(particle_size)))
        qsca = outputs[1][:, 0].reshape((len(wavelength), len(particle_size)))
        asym = outputs[2][:, 0].reshape((len(wavelength), len(particle_size)))

        return qext, qsca, asym

    def efficiencies(self, wavelength, particle_size, volume_mixing_ratios):
        """
        Calculate mie coefficients using mie python and LLL Approximation.

        Parameters
        ----------
        wavelength : np.ndarray or float
            Wavelength of the light [micron]
        particle_size : np.ndarray or float
            Size of the cloud particle [micron]
        volume_mixing_ratios : dict of np.ndarray or float
            Fraction of each cloud material given as float or array

        Return
        ------
        optical properties : np.ndarray
            extinction coefficient, scattering coefficient, and asymmetries parameter
        """
        # ==== Prepare inputs =============================================================

        # check inputs are correct type
        if not isinstance(wavelength, np.ndarray) and not isinstance(wavelength, (float, int)):
            print('Wavelength must be of type np.ndarray or float')
        if not isinstance(particle_size, np.ndarray) and not isinstance(particle_size, (float, int)):
            print('Particle size must be of type np.ndarray or float')
        if not isinstance(volume_mixing_ratios, dict) and not isinstance(volume_mixing_ratios, (float, int)):
            print('Volume mixing ratio must be of type dict or float')

        # convert floats to arrays
        if isinstance(wavelength, (float, int)):
            wavelength = np.array([wavelength])
        if isinstance(particle_size, (float, int)):
            particle_size = np.array([particle_size])

        # convert volume mixing ratio to array
        vmr = np.array(list(volume_mixing_ratios.values())).T

        # check vmr is the correct input shape
        if len(set(map(len, volume_mixing_ratios.values()))) != 1:
            print('Volume mixing ratios must have same shape')
        if len(particle_size) != len(vmr):
            print('Particle size and volume mixing ratio must have same shape')

        vmr_sum = np.sum(vmr, axis=1)
        if any(vmr_sum != 1):
            print('Volume mixing ratios do not add up to 1. The ratios have been renormalized.')
        idx = np.asarray(np.where(vmr_sum != 1))
        for i in idx[0]:
            vmr[i] = vmr[i] / sum(vmr[i])

        # ==== Load data for each species from files and get refractive index =======================================================

        #prepare output
        ref_index = np.zeros((len(self.species_list), len(wavelength), 2))
        for s, species in enumerate(self.species_list):

            # ==== Load data from files =============================================================================================

            # find species in files
            for file in self.files:
                if species in file:
                    # get data
                    content = open(file, 'r').readlines()
                    header = content[0].split()[1]
                    data = content[5:]

            # ==== Get the real(n) and imaginary (k) refractory index ===============================================================

            # loop over all wavelengths
            for wav, wave in enumerate(wavelength):
                # if desired wavelength is smaller than data, use the smallest wavelength data available
                if wave < float(data[0].split()[0]):
                    ref_index[s, wav, 0] = float(data[0].split()[1])
                    ref_index[s, wav, 1] = float(data[0].split()[2])
                    continue

                # if wavelength is within range log-log interpolation for dnr, _ in enumerate(data):
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
                        ref_index[s, wav, 0] = np.exp(np.log(nlo) + fac * np.log(nhi / nlo))
                        if klo <= 0 or khi <= 0:
                            ref_index[s, wav, 1] = 0
                        else:
                            ref_index[s, wav, 1] = np.exp(np.log(klo) + fac * np.log(khi / klo))

                        break

                else:
                    # if wavelength is out of range, extrapolate
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
                        ref_index[s, wav, 0] = np.exp(np.log(nhi) + fac * np.log(nlo / nhi))
                        if klo <= 0 or khi <= 0:
                            ref_index[s, wav, 1] = 0
                        else:
                            ref_index[s, wav, 1] = np.exp(np.log(khi) + fac * np.log(klo / khi))
                    else:
                        # non-conducting interpolation, linear decreasing k, constant n
                        ref_index[s, wav, 0] = float(data[-1].split()[1])
                        ref_index[s, wav, 1] = float(data[-1].split()[2]) * float(data[-1].split()[0]) / wave

        # ==== Combination of all wavelengths and particle size ====================================================
        final_wavelength = np.repeat(wavelength, len(particle_size))
        final_particle_size = np.tile(particle_size, len(wavelength))
        final_vmr = np.tile(vmr, (len(wavelength), 1))
        final_ref_index = np.repeat(ref_index, len(particle_size), axis=1)

        # prepare outputs
        mixed_ref_index = np.zeros(len(final_wavelength), dtype=complex)

        # ==== Find mixed refractive index ==========================================================================
        # loop over wavelength
        for wav, wave in enumerate(final_wavelength):
            # dielectric constant = (n + ik)^2
            ind_eff = (final_ref_index[:, wav, 0] + (1j * final_ref_index[:, wav, 1])) ** 2
            e_eff = (np.sum(final_vmr[wav] * (ind_eff ** (1 / 3)))) ** 3

            # find mixed refractive index, real(n) and imaginary(k)
            m_eff = np.sqrt(e_eff)
            mixed_ref_index[wav] = complex(m_eff.real, -m_eff.imag)

        # ==== Calculate Mie Efficiencies ====================================================================
        size_param = (2.0 * np.pi * final_particle_size) / final_wavelength

        # qe_temp = extinction, qs_temp = scattering, g_temp = asymmetry
        qe_temp, qs_temp, _, g_temp = mie.efficiencies_mx(mixed_ref_index, size_param)

        qext = qe_temp.reshape((len(wavelength), len(particle_size)))
        qsca = qs_temp.reshape((len(wavelength), len(particle_size)))
        asym = g_temp.reshape((len(wavelength), len(particle_size)))

        return qext, qsca, asym