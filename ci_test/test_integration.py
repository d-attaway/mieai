""" Integration tests """
import os
import unittest

import numpy as np

from mieai import Mieai


def test_ai():
    # ==== Set up
    ma = Mieai(default_model_location='files/')
    test_vars = ['qext', 'qsca', 'asym', 'wavelength']

    # ==== Standard Ai run
    extinction, scattering, asymmetry = ma.ai_efficiencies(
        np.logspace(-0.5, 1, 8), np.logspace(1.1, 1.9, 8),
        {
            'MgSiO3': np.linspace(0, 1, 8),
            'Fe': np.linspace(1, 0, 8),
            'TiO2': np.linspace(1, 0, 8)
        }
    )
    assert np.isclose(np.sum(extinction), 135.84726572036743)
    assert np.isclose(np.sum(scattering), 110.76739168167114)
    assert np.isclose(np.sum(asymmetry), 41.40172505378723)

    # ==== Use load grid model
    ma = Mieai(default_model_location='files/', load_ai_model='MODEL2')
    extinction, scattering, asymmetry = ma.ai_efficiencies(
        np.logspace(-0.5, 1, 8), np.logspace(1.1, 1.9, 8),
        {
            'MgSiO3': np.linspace(0, 1, 8),
            'Fe': np.linspace(1, 0, 8),
            'TiO2': np.linspace(1, 0, 8)
        }
    )
    assert np.isclose(np.sum(extinction), 135.84726572036743)
    assert np.isclose(np.sum(scattering), 110.76739168167114)
    assert np.isclose(np.sum(asymmetry), 41.40172505378723)

    # ==== Test float input
    extinction, scattering, asymmetry = ma.ai_efficiencies(3, 1,
        {
            'MgSiO3': [0.4],
            'Fe': [0.4],
            'TiO2': [0.4]
        }
    )
    assert np.isclose(np.sum(extinction), 2.857503890991211)
    assert np.isclose(np.sum(scattering), 2.3773093223571777)
    assert np.isclose(np.sum(asymmetry), 0.39067474007606506)

    # ==== Test wrong model load
    testcase = unittest.TestCase()
    with testcase.assertRaises(ValueError):
        ma = Mieai(default_model_location='files/', load_ai_model='NON_EXISTING')

    # ==== Test non-initialisation error
    with testcase.assertRaises(ValueError):
        ma = Mieai(use_ai=False)
        _, _, _ = ma.ai_efficiencies(None, None, None)


def test_grid():
    # ==== Set up
    ma = Mieai(use_ai=False, mute=False)
    test_vars = ['qext', 'qsca', 'asym', 'wavelength']

    # ==== create tiny grid
    ds = ma.produce_efficiency_grid(
        ['SiO2', 'Fe'], wavelengths=np.logspace(-1 ,1.3 ,10),
        particle_sizes=np.logspace(1,2 ,40), vmr_data_points=4,
        save_file='grid_test.nc'
    )
    expected_vals = [3373.4240545112407, 2735.199995088339, 1124.5631941617262,
                     44.7320528831799]
    for t, test in enumerate(test_vars):
        assert np.isclose(np.sum(ds[test]), expected_vals[t])

    # ==== read in grid
    ma.load_grid_efficiency(file_name='grid_test.nc')
    lo = ma.default_grids['grid_test.nc']['ds']
    for t, test in enumerate(test_vars):
        assert np.isclose(np.sum(lo[test]), expected_vals[t])
    assert ['SiO2', 'Fe'] == lo.attrs['species']
    testcase = unittest.TestCase()
    with testcase.assertRaises(ValueError):
        ma.load_grid_efficiency(file_name='grid_that_does_not_exist.nc')

    # === use grid evaluation
    extinction, scattering, asymmetry = ma.grid_efficiencies(
        np.logspace(-0.5, 1, 8), np.logspace(1.1, 1.9, 8),
        {'SiO2': np.linspace(0, 1, 8), 'Fe': np.linspace(1, 0, 8)}
    )
    assert np.isclose(np.sum(extinction), 134.77024714311165)
    assert np.isclose(np.sum(scattering), 108.05183641939178)
    assert np.isclose(np.sum(asymmetry), 45.20338788423523)

    # === use grid evaluation with imediate read in
    extinction, scattering, asymmetry = ma.grid_efficiencies(
        np.logspace(-0.5, 1, 8), np.logspace(1.1, 1.9, 8),
        {'SiO2': np.linspace(0, 1, 8), 'Fe': np.linspace(1, 0, 8)},
        grid_file='grid_test.nc'
    )
    assert np.isclose(np.sum(extinction), 134.77024714311165)
    assert np.isclose(np.sum(scattering), 108.05183641939178)
    assert np.isclose(np.sum(asymmetry), 45.20338788423523)

    # ==== request species that are not available
    with testcase.assertRaises(ValueError):
        extinction, scattering, asymmetry = ma.grid_efficiencies(
            np.logspace(-0.5, 1, 8), np.logspace(1.1, 1.9, 8),
            {'Not': np.linspace(0, 1, 8), 'Exist': np.linspace(1, 0, 8)},
        )

    # ==== finish up
    os.remove('grid_test.nc')
