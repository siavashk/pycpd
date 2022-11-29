from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pycpd import ConstrainedDeformableRegistration
import numpy as np

fish_source = np.loadtxt('data/fish_source.txt')
marker_size = 100
N_pts_include = 61
IDs = [1,10,20,30]
IDs_Y = IDs + [fish_source.shape[0] + i for i in IDs]
IDs_X = IDs + [N_pts_include + i for i in IDs]

def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ids_X = np.arange(0, X.shape[0])
    ids_X = np.delete(ids_X, IDs_X)

    ids_Y = np.arange(0, Y.shape[0])
    ids_Y = np.delete(ids_Y, IDs_Y)


    ax.scatter(X[ids_X, 0],  X[ids_X, 1], X[ids_X, 2], color='red', label='Target')
    ax.scatter(Y[ids_Y, 0],  Y[ids_Y, 1], Y[ids_Y, 2], color='blue', label='Source')

    ax.scatter(X[IDs_X, 0],  X[IDs_X, 1], X[IDs_X, 2], color='red', label='Target Constrained', s=marker_size, facecolors='none')
    ax.scatter(Y[IDs_Y, 0],  Y[IDs_Y, 1], Y[IDs_Y, 2], color='green', label='Source Constrained', s=marker_size, marker=(5, 1))
    
    ax.text2D(0.87, 0.92, 'Iteration: {:d}'.format(
        iteration), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)


def main():
    fish_target = np.loadtxt('data/fish_target.txt')
    
    #simulate a pointcloud missing certain parts
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
    # select fixed correspondences
    src_id = np.int32(IDs_Y)
    tgt_id = np.int32(IDs_X)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    callback = partial(visualize, ax=ax)

    reg = ConstrainedDeformableRegistration(**{'X': X, 'Y': Y, 'low_rank': True}, e_alpha = 1e-8, source_id = src_id, target_id = tgt_id)
    reg.register(callback)
    plt.show()


if __name__ == '__main__':
    main()
