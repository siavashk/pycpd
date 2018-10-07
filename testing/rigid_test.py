import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from pycpd import rigid_registration

def test_2D():
    theta = np.pi / 6.0
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    t = np.array([0.5, 1.0])

    Y = np.loadtxt('data/fish_target.txt')
    X = np.dot(Y, R) + np.tile(t, (np.shape(Y)[0], 1))

    reg = rigid_registration(**{ 'X': X, 'Y':Y })
    TY, (s_reg, R_reg, t_reg) = reg.register()
    assert_almost_equal(1.0, s_reg)
    assert_array_almost_equal(R, R_reg)
    assert_array_almost_equal(t, t_reg)
    assert_array_almost_equal(X, TY)

def test_3D():
    theta = np.pi / 6.0
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    t = np.array([0.5, 1.0, -2.0])

    fish_target = np.loadtxt('data/fish_target.txt')
    Y = np.zeros((fish_target.shape[0], fish_target.shape[1] + 1))
    Y[:,:-1] = fish_target
    X = np.dot(Y, R) + np.tile(t, (np.shape(Y)[0], 1))

    reg = rigid_registration(**{ 'X': X, 'Y':Y })
    TY, (s_reg, R_reg, t_reg) = reg.register()
    assert_almost_equal(1.0, s_reg)
    assert_array_almost_equal(R, R_reg)
    assert_array_almost_equal(t, t_reg)
    assert_array_almost_equal(X, TY)
