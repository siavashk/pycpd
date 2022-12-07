import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from pycpd import gaussian_kernel, ConstrainedDeformableRegistration


def test_2D():
    X = np.loadtxt('data/fish_target.txt')
    Y = np.loadtxt('data/fish_source.txt')
    
    # simulate a pointcloud missing certain parts
    X = X[:61]
    # select fixed correspondences
    src_id = np.int32([1,10,20,30])
    tgt_id = np.int32([1,10,20,30])

    reg = ConstrainedDeformableRegistration(**{'X': X, 'Y': Y}, e_alpha = 1e-8, source_id = src_id, target_id = tgt_id)  
    TY, _ = reg.register()
    assert_array_almost_equal(X[tgt_id], TY[src_id], decimal=1)


def test_3D():
    fish_target = np.loadtxt('data/fish_target.txt')
    fish_target = fish_target[:61]

    X1 = np.zeros((fish_target.shape[0], fish_target.shape[1] + 1))
    X1[:, :-1] = fish_target
    X2 = np.ones((fish_target.shape[0], fish_target.shape[1] + 1))
    X2[:, :-1] = fish_target
    X = np.vstack((X1, X2))

    fish_source = np.loadtxt('data/fish_source.txt')
    Y1 = np.zeros((fish_source.shape[0], fish_source.shape[1] + 1))
    Y1[:, :-1] = fish_source
    Y2 = np.ones((fish_source.shape[0], fish_source.shape[1] + 1))
    Y2[:, :-1] = fish_source
    Y = np.vstack((Y1, Y2))
    
    # select fixed correspondences
    src_id = np.int32([1,10,20,30, len(Y1)+1, len(Y1)+10, len(Y1)+20, len(Y1)+30])
    tgt_id = np.int32([1,10,20,30, len(X1)+1, len(X1)+10, len(X1)+20, len(X1)+30])


    reg = ConstrainedDeformableRegistration(**{'X': X, 'Y': Y}, e_alpha = 1e-8, source_id = src_id, target_id = tgt_id)
    TY, _ = reg.register()
    assert_array_almost_equal(TY[src_id], X[tgt_id], decimal=0)


def test_3D_low_rank():
    fish_target = np.loadtxt('data/fish_target.txt')
    fish_target = fish_target[:61]

    X1 = np.zeros((fish_target.shape[0], fish_target.shape[1] + 1))
    X1[:, :-1] = fish_target
    X2 = np.ones((fish_target.shape[0], fish_target.shape[1] + 1))
    X2[:, :-1] = fish_target
    X = np.vstack((X1, X2))

    fish_source = np.loadtxt('data/fish_source.txt')
    Y1 = np.zeros((fish_source.shape[0], fish_source.shape[1] + 1))
    Y1[:, :-1] = fish_source
    Y2 = np.ones((fish_source.shape[0], fish_source.shape[1] + 1))
    Y2[:, :-1] = fish_source
    Y = np.vstack((Y1, Y2))

    # select fixed correspondences
    src_id = np.int32([1,10,20,30, len(Y1)+1, len(Y1)+10, len(Y1)+20, len(Y1)+30])
    tgt_id = np.int32([1,10,20,30, len(X1)+1, len(X1)+10, len(X1)+20, len(X1)+30])

    reg = ConstrainedDeformableRegistration(**{'X': X, 'Y': Y, 'low_rank': True}, e_alpha = 1e-8, source_id = src_id, target_id = tgt_id)
    TY, _ = reg.register()
    assert_array_almost_equal(TY[src_id], X[tgt_id], decimal=0)

