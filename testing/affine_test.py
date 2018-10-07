import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from pycpd import affine_registration

def test_2D():
    B = np.array([[1.0, 0.5], [0, 1.0]])
    t = np.array([0.5, 1.0])

    Y = np.loadtxt('data/fish_target.txt')
    X = np.dot(Y, B) + np.tile(t, (np.shape(Y)[0], 1))

    reg = affine_registration(**{ 'X': X, 'Y':Y })
    TY, (B_reg, t_reg) = reg.register()
    assert_array_almost_equal(B, B_reg)
    assert_array_almost_equal(t, t_reg)
    assert_array_almost_equal(X, TY)

def test_3D():
    B = np.array([[1.0, 0.5, 0.0], [0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    t = np.array([0.5, 1.0, -2.0])

    fish_target = np.loadtxt('data/fish_target.txt')
    Y1 = np.zeros((fish_target.shape[0], fish_target.shape[1] + 1))
    Y1[:,:-1] = fish_target
    Y2 = np.ones((fish_target.shape[0], fish_target.shape[1] + 1))
    Y2[:,:-1] = fish_target
    Y = np.vstack((Y1, Y2))

    X = np.dot(Y, B) + np.tile(t, (np.shape(Y)[0], 1))

    reg = affine_registration(**{ 'X': X, 'Y':Y })
    TY, (B_reg, t_reg) = reg.register()
    assert_array_almost_equal(B, B_reg)
    assert_array_almost_equal(t, t_reg)
    assert_array_almost_equal(X, TY)
