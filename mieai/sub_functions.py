""" General functionalities """
# pylint: disable=C0415,R0902,R0912,R0914,R0915

import os
import pandas as pd
import numpy as np
import yaml

def read_in_refindex(species, wavelength, files):
    """
    Read in and interpolate refractive index files.

    Parameters
    ----------
    species : List with size N
        Name of cloud species.
        wavelength : np.ndarray or float of size M
            Wavelength of the light [micron]
    files : List
        Refractive index files

    Return
    ------
    ref_index : np.ndarray of size (N, M, 2)
        Refractive index data: real, and imaginary part.
    """

    # prepare output
    ref_index = np.zeros((len(species), len(wavelength), 2))
    for s, spec in enumerate(species):

        # ==== Load data from files =====================================================

        # find species in files
        data = None
        for file in files:
            if spec in file:
                # get data using pandas
                content = pd.read_csv(file, sep=r'\s+', header=None, usecols=[1, 2, 3])
                # convert to array and flip vertically so wavelength increases
                data = np.flip(content.to_numpy(), axis=0)
        if data is None:
            raise ValueError('No refindex file found for ' + spec)

        # ==== Get the real(n) and imaginary (k) refractory index =======================
        # prepare output
        if not isinstance(wavelength, np.ndarray):
            wavelength = np.asarray([wavelength])

        # loop over all wavelengths
        for wav, wave in enumerate(wavelength):
            # if desired wavelength is smaller than data, use the smallest wavelength
            # data available
            if wave < float(data[0, 0]):
                ref_index[s, wav, 0] = float(data[0, 1])
                ref_index[s, wav, 1] = float(data[0, 2])
                continue

            # if wavelength is within range log-log interpolation
            for dnr, _ in enumerate(data):
                cur_wave = float(data[dnr, 0])  # current wavelength
                if wave < cur_wave:
                    nlo = float(data[dnr - 1, 1])  # lower n value
                    nhi = float(data[dnr, 1])  # higher n value
                    klo = float(data[dnr - 1, 2])  # lower k value
                    khi = float(data[dnr, 2])  # higher k value
                    prev_wave = float(data[dnr - 1, 0])  # previous wavelength
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
                # non-conducting interpolation, linear decreasing k, constant n
                ref_index[s, wav, 0] = float(data[-1, 1])
                ref_index[s, wav, 1] = float(data[-1, 2]) * float(data[-1, 0]) / wave

    return ref_index


def calculate_subradii(particle_size, vmr):
    """
    Calculate subgrid for each radius.

    Parameters
    ----------
    particle_size : np.ndarray or float of size M
        Size of the cloud particle [micron]
    vmr : ndarray
        Fraction of each cloud material

    Return
    ------
    sub_rad, vmr : (ndarray(M*6), ndarray)
        Sub-spacing of radii and adjusted vmr.
    """
    if len(particle_size) > 1:
        if len(set(particle_size)) != 1:
            # prepare outputs
            rad_min = np.zeros_like(particle_size)
            rad_max = np.zeros_like(particle_size)
            mid_points = (particle_size[1:] + particle_size[:-1]) / 2

            # radius minimum and maximum from midpoints
            rad_min[1:] = mid_points
            # smallest radius value >0
            rad_min[0] = np.max([particle_size[0] - mid_points[0], 0])
            rad_max[:-1] = mid_points
            rad_max[-1] = particle_size[-1] + mid_points[-1]

            # prepare output
            sub_rad = np.zeros((len(particle_size) * 6))
            i = 0  # index

            for r_max, r_min in zip(rad_max, rad_min):
                # six radius points to average over
                r = (r_max - r_min) / 6
                rad_range = r_min + np.array([r, 2 * r, 3 * r, 4 * r, 5 * r, 6 * r])
                sub_rad[i:i + 6] = rad_range
                # index
                i += 6
            # make volume mixing ratios the same size as particle size
            vmr = np.repeat(vmr, 6, axis=0)

        else:
            sub_rad = np.zeros((len(particle_size) * 6))
            i = 0
            for rad in particle_size:
                sub_rad[i:i + 6] = np.linspace(rad * 0.7, rad * 1.3, 6)
                i += 6
            vmr = np.repeat(vmr, 6, axis=0)

    else:
        sub_rad = particle_size

    return sub_rad, vmr

def get_model_info(model_name):
    '''
    Get neural network files and information.

    Parameters
    ----------
    model_name: string

    Returns
    -------
    List of files that correspond to the network.

    '''
    config_yaml = os.path.dirname(__file__) + '/config.yaml'
    with open(config_yaml, 'r') as f:
        config = yaml.safe_load(f)

    try:
        model_info = config[model_name]
        files = model_info['files']
        low_wave = model_info['low_wave']
        high_wave = model_info['high_wave']
        return files, low_wave, high_wave
    except KeyError:
        raise ValueError(f"Network '{model_name}' not found in config")

def initialize_ai_models(load_ai_model, model_names, model_path):
    '''
    Load ai tensorflow models.

    Parameters
    ----------
    load_model : String
        Either 'all' to load every model or the model name
    model_names: Dictionary
        Dictionary of species in each mixture

    Returns
    ----------
    low_waves, high_waves: Dictionaries
        Lower/higher wavelength cutoffs for each mixture (returned for 'all')
    low_models, mid_models, high_models: Dictionaries
        Loaded models in each wavelength range for each mixture (returned for 'all')
    low_wave, high_wave: floats
        Lower/higher wavelength cutoffs (returned for a specified model)
    low_model, mid_model, high_model: tensorflow models
        Loaded models in each wavelength range (returned for a specified model)
    best_model: tuple
        The specified mixture and its species (returned for a specified model)
    '''
    from tensorflow.keras.models import load_model

    # load all models by default if one is not specified
    if load_ai_model == 'all':

        # prepare output
        low_waves = {}
        high_waves = {}
        low_models = {}
        mid_models = {}
        high_models = {}

        for model in model_names.keys():
            (model_files, low_waves[model], high_waves[model]) = get_model_info(model)

            # load all models for each mixture
            low_models[model] = load_model(
                model_path + model_files[0]
            )
            mid_models[model] = load_model(
                model_path + model_files[1]
            )
            high_models[model] = load_model(
                model_path + model_files[2]
            )

        return low_waves, high_waves, low_models, mid_models, high_models

    else:
        # get info for the specified model
        model_files, low_wave, high_wave \
            = get_model_info(load_ai_model)

        # save mixture info
        best_model = (load_ai_model, model_names[load_ai_model])

        # load models for each wavelength range
        low_model = load_model(
            model_path + model_files[0]
        )
        mid_model = load_model(
            model_path + model_files[1]
        )
        high_model = load_model(
            model_path + model_files[2]
        )

        return low_wave, high_wave, low_model, mid_model, high_model, best_model