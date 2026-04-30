""" This file contains all functionaliteis to use the pre-calculated grids """
import os
import numpy as np
import xarray as xr
from glob import glob
from time import time
from datetime import datetime, timedelta

def grid_efficiencies(self, wavelength, particle_size, volume_mixing_ratios,
                      grid_file=None):
    """
    Approximate mie coefficients using mie python and LLL Approximation read in from
    the grid_file.

    Parameters
    ----------
    wavelength : np.ndarray or float of size N
        Wavelength of the light [micron]
    particle_size : np.ndarray or float of size M
        Size of the cloud particle [micron]
    volume_mixing_ratios : dict of np.ndarray or float of size M for each species
        Fraction of each cloud material given as float or array
    grid_file : string
        Path to the grid file.

    Return
    ------
    optical properties : np.ndarray of size (M, N)
        extinction coefficient, scattering coefficient, and asymmetries parameter
    """

    # ==== Load grid
    if grid_file is None:
        # select mixing theory
        grid = self.default_grids
        # find all dataset that include all species
        L_set = set(volume_mixing_ratios.keys())
        valid_datasets = {
            name: data['species'] for name, data in grid.items()
            if L_set.issubset(data['species'])
        }
        # check if there are no matching grids
        if not valid_datasets:
            raise ValueError("No default grid for " + str(L_set) +
                             " is available. Please provide one via grid_file.")
        # Now pick the dataset with the smallest total size
        best_dataset = min(valid_datasets.items(), key=lambda item: len(item[1]))
        # open that dataset
        ds = grid[best_dataset[0]]['ds']
    else:
        # ==== check data grid
        for specs in ds.attrs['species']:
            if specs not in volume_mixing_ratios:
                raise ValueError("The selected grid requires the volume mixing "
                                 "ratio of " + specs)
        ds = xr.open_dataset(grid_file, engine="h5netcdf")


    # ==== read out data
    # define arguments for interpolation from xarray
    args = {
        'wavelength': wavelength,
        "particle_size": ("points", particle_size),
        'method': 'linear'
    }
    # loop over all species
    for spec in ds.attrs['species']:
        # skip implicit species
        if spec == ds.attrs['implicit_species']: continue
        # if the species is given, use the vmr
        if spec in volume_mixing_ratios:
            # add non-implicit species
            args['VMR_' + spec] = ("points", volume_mixing_ratios[spec])
        # if the species is not given, set it to 0
        else:
            args['VMR_' + spec] = ("points", np.zeros(len(particle_size)))


    # interpolate from xarray
    extinction = np.nan_to_num(ds['qext'].interp(**args))
    scattering = np.nan_to_num(ds['qsca'].interp(**args))
    asymmetry = np.nan_to_num(ds['asym'].interp(**args))

    return extinction, scattering, asymmetry


def produce_efficiency_grid(self, species, wavelengths=np.logspace(-1 ,1.3 ,200),
                            particle_sizes=np.logspace(-4 ,3.1 ,100), vmr_data_points=20,
                            theory='LLL', save_file=None):
    """
    Calculate mie coefficient grid using mie python and LLL Approximation.

    Parameters
    ----------
    species : List
        Species names
    wavelengths : np.ndarray or float of size N
        Wavelength of the light [micron]
    particle_sizes : np.ndarray or float of size M
        Size of the cloud particle [micron]
    vmr_data_points : int
        Number of volume fraction mixing ratio points
    theory : str, optional
        Mixing theory used, can either be 'LLL' (Default) or 'Burggeman'
    save_file : str
        Path to save the grid file

    Return
    ------
    ds : xarray.DataSet
        Data set containing the extinction coefficient, scattering coefficient, and asymmetries parameter
    """

    # ==== get shape of output array and prepare coordinates of dataset
    shape = [len(particle_sizes), len(wavelengths)]  # shape of data array
    dims = ['particle_size', 'wavelength']  # name of dimensions
    vmrs = np.linspace(0, 1, vmr_data_points)
    vmr = {}  # prepare standard vmr array
    vmr_array = np.ones(len(particle_sizes))
    coords = {
        'particle_size': particle_sizes,
        'wavelength': wavelengths,
    }
    for _, spec in enumerate(species):
        vmr[spec] = vmr_array.copy
    # ==== adatpitve fill in for species, last on is implicit
    for _, spec in enumerate(species[:-1]):
        shape.append(vmr_data_points)
        dims.append('VMR_' + spec)
        coords['VMR_' + spec] = np.linspace(0, 1, vmr_data_points)
    # ==== data array
    qext = np.zeros(shape)
    qsca = np.zeros(shape)
    asym = np.zeros(shape)

    # ==== get indexing for vmrs
    arrays = [np.arange(vmr_data_points) for _ in range(len(species ) -1)]
    grids = np.meshgrid(*arrays, indexing='ij')
    vmr_index = np.stack(grids, axis=-1).reshape(-1, len(species ) -1)

    # ==== Fill in Grid
    start_time = time()
    for v, vmri in enumerate(vmr_index):
        if v > 0:
            dt = (time() - start_time ) / v *(len(vmr_index ) -v)
            now = datetime.fromtimestamp(time())
            eta = now + timedelta(seconds=dt)
            eta = eta.strftime("%Y-%m-%d %H:%M:%S")
        else:
            eta = '--'
        print(f'Progress: { v /len(vmr_index ) *100:.1f}% (ETA: {eta})')
        vmr_last = 0
        vmr = {}
        for s, spec in enumerate(species[:-1]):
            vmr[spec] = vmrs[vmri[s] ] *vmr_array.copy()
            vmr_last += vmrs[vmri[s]]
        vmr[species[-1]] = np.max([1 - vmr_last, 0] ) *vmr_array.copy()
        line = self.efficiencies(wavelengths, particle_sizes, vmr, theory=theory)
        qext[:, :, *vmri] = line[0]
        qsca[:, :, *vmri] = line[1]
        asym[:, :, *vmri] = line[2]

    # ==== Generate dataset
    ds = xr.Dataset(
        data_vars={
            'qext': (dims, qext),
            'qsca': (dims, qsca),
            'asym': (dims, asym),
        },
        coords=coords,
        attrs={
            'species': species,
            'implicit_species': species[-1],
        }
    )

    # ==== Save dataset if a save file is given
    if save_file is not None:
        ds.to_netcdf(save_file, engine="h5netcdf")

    return ds


def load_grid_efficiency(self, file_name=None):
    """
    Load previously calculated opacity grids

    Parameters
    ----------
    self : MieAi class
    file_name : If given, only this file is loaded, if None, data_path will be checked.
    """

    # ==== Check if only one file should be loaded, or all files from data_path
    if file_name is None:
        grid_files = glob(self.data_path + 'grid_*.nc')
    else:
        grid_files = [file_name]

    # ==== Looop over all files
    for grid_file in grid_files:
        try:
            # get data and assign it to the dictionary
            ds = xr.open_dataset(grid_file, engine="h5netcdf")
            self.default_grids[grid_file] = {'species': ds.attrs['species'], 'ds': ds}
            if not self.mute:
                print('[INFO] Added grid for', ds.attrs['species'], f"from {round(ds['wavelength'].values[0], 2)} to {round(ds['wavelength'].values[-1], 2)} micron.")
            ds.close()
        except:
            # this error only rises if the file loaded is not what was expected.
            if not self.mute:
                print('[WARN] The following grid file could not be loded:\n    ', grid_file)
