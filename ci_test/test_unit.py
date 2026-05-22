""" Integration tests """
import numpy as np
from mieai import Mieai

def test_sub_functions():
    # ==== test Bruggeman
    ma = Mieai(use_ai=False, mute=False)
    extinction, scattering, asymmetry = ma.efficiencies(
        np.logspace(-0.5, 1, 8), np.logspace(1.1, 1.9, 8),
        {'SiO2': np.linspace(0, 1, 8), 'Fe': np.linspace(1, 0, 8)},
        theory='Bruggeman'
    )
    assert np.isclose(np.sum(extinction), 135.02057908318517)
    assert np.isclose(np.sum(scattering), 107.51411053800598)
    assert np.isclose(np.sum(asymmetry), 44.00588855065756)

