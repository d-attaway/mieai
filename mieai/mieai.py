import os
import numpy as np
from tensorflow import keras
from keras import layers
import glob
import miepython as mie
import pandas as pd

from .sub_functions import read_in_refindex, calculate_subradii
from .mixing_theory import mixing_theory


class Mieai:
    def __init__(self, use_ai=True):

        # ==== General preparations ======================================================================
        # Load species data from files
        self.files = glob.glob(os.path.dirname(__file__) + '/opacity_files/*.refrind')
        self.available_species = [os.path.basename(path).split('/')[0][:-8] for path in self.files]
        # save ai initialisation state
        self.use_ai = use_ai

        # ==== Prepare Neural Network ====================================================================
        if use_ai:
            # define sepcies list TODO: allow for more combinations of species
            self.species_list = ['TiO2', 'Fe', 'Mg2SiO4']
            # ==== Define ML model =======================================================================
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

    def ai_efficiencies(self, wavelength, particle_size, volume_mixing_ratios):
        """
        Calculate mie coefficients using a pre-trained neural network.

        Parameters
        ----------
        wavelength : np.ndarray or float of size N
            Wavelength of the light [micron]
        particle_size : np.ndarray or float of size M
            Size of the cloud particle [micron]
        volume_mixing_ratios : dict of np.ndarray or float of size M for each species
            Fraction of each cloud material given as float or array

        Return
        ------
        optical properties : np.ndarray of size (M, N)
            extinction coefficient, scattering coefficient, and asymmetries parameter
        """

        # ==== Input checks =============================================================

        # check if neural network is initalised
        if not self.use_ai:
            raise

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
        wavelength : np.ndarray or float of size N
            Wavelength of the light [micron]
        particle_size : np.ndarray or float of size M
            Size of the cloud particle [micron]
        volume_mixing_ratios : dict of np.ndarray or float of size M for each species
            Fraction of each cloud material given as float or array

        Return
        ------
        optical properties : np.ndarray of size (M, N)
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

        # define species list according to entries in vmr
        species_list = list(volume_mixing_ratios.keys())

        # check if all species are available
        for spec in species_list:
            if spec not in self.available_species:
                raise ValueError("The species " + spec + " is not available")

        # create array with vmr values
        vmr = np.zeros((len(particle_size), len(species_list)))
        for s, spec in enumerate(species_list):
            vmr[:, s] = volume_mixing_ratios[spec]

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

        # ==== Radius averaging =============================================================================
        sub_rad, vmr = calculate_subradii(particle_size, vmr)

        # ==== Load data for each species from files and get refractive index ===============================
        ref_index = read_in_refindex(species_list, wavelength, self.files)

        # ==== Combination of all wavelengths and particle size =============================================
        final_wavelength = np.repeat(wavelength, len(sub_rad))
        final_sub_rad = np.tile(sub_rad, len(wavelength))
        final_vmr = np.tile(vmr, (len(wavelength), 1))
        final_ref_index = np.repeat(ref_index, len(sub_rad), axis=1)

        mixed_ref_index = mixing_theory(final_wavelength, final_ref_index, final_vmr)

        # ==== Calculate Mie Efficiencies ====================================================================
        size_param = (2.0 * np.pi * final_sub_rad) / final_wavelength

        # qe_temp = extinction, qs_temp = scattering, g_temp = asymmetry
        qe_temp, qs_temp, _, g_temp = mie.efficiencies_mx(mixed_ref_index, size_param)

        # ==== Prepare outputs ==============================================================================
        if len(sub_rad) != len(particle_size):
            extinction = np.mean(qe_temp.reshape(len(particle_size) * len(wavelength), 6), axis=1).reshape(len(wavelength), len(particle_size)).T
            scattering = np.mean(qs_temp.reshape(len(particle_size) * len(wavelength), 6), axis=1).reshape(len(wavelength), len(particle_size)).T
            asymmetry = np.mean(g_temp.reshape(len(particle_size) * len(wavelength), 6), axis=1).reshape(len(wavelength), len(particle_size)).T

        else:
            extinction = qe_temp.reshape(len(wavelength), len(particle_size)).T
            scattering = qs_temp.reshape(len(wavelength), len(particle_size)).T
            asymmetry = g_temp.reshape(len(wavelength), len(particle_size)).T

        return extinction, scattering, asymmetry
