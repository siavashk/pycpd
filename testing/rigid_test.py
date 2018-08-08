import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from pycpd import rigid_registration

def test_2D():
    theta = np.pi / 6.0
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    t = np.array([0.5, 1.0])

    X = np.loadtxt('data/fish_target.txt')
    Y = np.dot(X, R) + np.tile(t, (np.shape(X)[0], 1))

    reg = rigid_registration(**{ 'X': X, 'Y':Y })
    TY, _ = reg.register()
    assert_array_almost_equal(TY, X)
