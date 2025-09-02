import os
import numpy as np
from tensorflow import keras
from keras import layers


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

    def efficiencies(self, wavelength, particle_size, volume_mixing_ratios):
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
        # ==== TODO: check if inputs are floats or arrays, if arrays check dimensions

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
        final_vmr = np.tile(vmr, (len(wavelength), 1))

        # check dimensions
        if len(set(map(len, volume_mixing_ratios.values()))) != 1:
            print('Volume mixing ratios must have same shape')
        if len(particle_size) != len(vmr):
            print('Wavelength, particle size, and volume mixing ratio must have same shape')

        # ==== TODO: check if vmrs add to 1, give warning and renormalize if not

        vmr_sum = np.sum(final_vmr, axis=1)
        if any(vmr_sum != 1):
            print('Volume mixing ratios do not add up to 1. The ratios have been renormalized.')
        idx = np.asarray(np.where(vmr_sum != 1))
        for i in idx[0]:
            final_vmr[i] = final_vmr[i] / sum(final_vmr[i])

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


