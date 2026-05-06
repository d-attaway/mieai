import os
import numpy as np
import glob
import xarray as xr
import miepython as mie

from tensorflow.keras.models import load_model # figure out how to load in with use_ai?
from time import time
from datetime import datetime, timedelta

from .sub_functions import read_in_refindex, calculate_subradii, get_model_info
from .mixing_theory import mixing_theory


class Mieai:

    # ==== Import functions from sub-files ========================================================
    from .grid import grid_efficiencies, produce_efficiency_grid, load_grid_efficiency

    def __init__(self, use_ai=True, default_data_location=None, mute=True):
        """
        Constructor

        Parameters
        ----------
        use_ai : bool
            If False, AI will be disabled. This allows to use MieAi without installing tensorflow.
        default_data_location : str, optional
            Location of opacity data. If none, MieAi defaults are used.
        mute : bool, optional
            If True, MieAi will produce no diagnostic outputs and runs quietly.
        """

        # ==== General preparations ======================================================================
        # Load species data from files
        self.files = glob.glob(os.path.dirname(__file__) + '/opacity_files/*.refrind')
        self.available_species = [os.path.basename(path).split('/')[0][:-8] for path in self.files]
        # save ai initialisation state
        self.use_ai = use_ai
        # save mute preference
        self.mute = mute

        # ==== Prepare Neural Network ====================================================================
        if use_ai:
            # ==== Import tensorflow here so people can use Mieai without it
            from tensorflow.keras.models import load_model

        self.model_names = {
            "MODEL1": ['TiO2', 'Fe', 'Mg2SiO4'],
            "MODEL2": ['TiO2', 'Fe', 'MgSiO3'],
            "MODEL3": ['SiO2', 'MgSiO3', 'Mg2SiO4'],
            "MODEL4": ['SiO2', 'MgSiO3', 'Fe'],
        } # move with use_ai?

        # ==== List of default datasets
        # user input data location
        if default_data_location is not None:
            self.data_path = default_data_location
        # default data location
        else:
            self.data_path = os.path.dirname(__file__) + '/../data/'

        # ==== Load predetermined grid dataset
        # default datasets
        self.default_grids = {}
        self.load_grid_efficiency()


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

        # ==== network intialization & retrieval ==================================================

        # check if neural network is initalised
        if not self.use_ai:
            raise

        # find all models that include all species
        L_set = set(volume_mixing_ratios.keys())
        valid_models = {
            name: data for name, data in self.model_names.items()
            if L_set.issubset(data)
        }
        # check if there are no matching models
        if not valid_models:
            raise ValueError("No network for " + str(L_set) + " is available.")

        # Now pick the model with the smallest total size
        best_dataset = min(valid_models.items(), key=lambda item: len(item[1]))

        # get files for the model
        model_files, low_wave, high_wave = get_model_info(best_dataset[0])

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

        # reorder volume mixing ratios and turn into array
        vmr_arr = {key: volume_mixing_ratios[key] for key in best_dataset[1]}
        vmr = np.array(list(vmr_arr.values())).T

        if len(set(map(len, volume_mixing_ratios.values()))) != 1 and not self.mute:
            print('Volume mixing ratios must have same shape')

        vmr_sum = np.sum(vmr, axis=1)
        if any(vmr_sum != 1) and not self.mute:
            print('Volume mixing ratios do not add up to 1. The ratios have been renormalized.')
        idx = np.asarray(np.where(vmr_sum != 1))
        for i in idx[0]:
            vmr[i] = vmr[i] / sum(vmr[i])

        # make volume mixing ratios have the same dimensions as final wavelength & final particle size
        final_vmr = np.tile(vmr, (len(wavelength), 1))

        # ==== Prepare model ============================================================
        # stack inputs
        inputs = np.stack((np.asarray(np.log10(final_wavelength)),
                           np.asarray(np.log10(final_particle_size)),
                           np.asarray(final_vmr[:,0]),
                           np.asarray(final_vmr[:,1])), axis=1)

        # load models for each wavelength range
        low_model = load_model(os.path.dirname(__file__) + '/models/' + model_files[0])
        mid_model = load_model(os.path.dirname(__file__) + '/models/' + model_files[1])
        high_model = load_model(os.path.dirname(__file__) + '/models/' + model_files[2])

        # prepare output
        extinction = np.zeros((len(inputs), 1))
        scattering = np.zeros((len(inputs), 1))
        asymmetry = np.zeros((len(inputs), 1))

        # masks for each wavelength range
        low_mask = inputs[:, 0] <= np.log10(low_wave)
        mid_mask = ((inputs[:, 0] > np.log10(low_wave)) & (inputs[:, 0] < np.log10(high_wave)))
        high_mask = inputs[:, 0] >= np.log10(high_wave)

        # predict coefficients for each wavelength range
        extinction[low_mask], scattering[low_mask], asymmetry[low_mask] = low_model.predict(inputs[low_mask])
        extinction[mid_mask], scattering[mid_mask], asymmetry[mid_mask] = mid_model.predict(inputs[mid_mask])
        extinction[high_mask], scattering[high_mask], asymmetry[high_mask] = high_model.predict(inputs[high_mask])

        # reshape outputs
        qext = extinction[:, 0].reshape((len(wavelength), len(particle_size)))
        qsca = scattering[:, 0].reshape((len(wavelength), len(particle_size)))
        asym = asymmetry[:, 0].reshape((len(wavelength), len(particle_size)))

        return qext, qsca, asym

    def efficiencies(self, wavelength, particle_size, volume_mixing_ratios, theory='LLL'):
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
        theory : str, optional
            Mixing theory used, can either be 'LLL' (Default) or 'Burggeman'

        Return
        ------
        optical properties : np.ndarray of size (M, N)
            extinction coefficient, scattering coefficient, and asymmetries parameter
        """
        # ==== Prepare inputs =============================================================

        # check inputs are correct type
        if not self.mute:
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
        if len(set(map(len, volume_mixing_ratios.values()))) != 1 and not self.mute:
            print('Volume mixing ratios must have same shape')
        if len(particle_size) != len(vmr):
            print('Particle size and volume mixing ratio must have same shape')

        vmr_sum = np.sum(vmr, axis=1)
        if any(vmr_sum != 1) and not self.mute:
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

        mixed_ref_index = mixing_theory(final_wavelength, final_ref_index, final_vmr, theory=theory)

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
