from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pycpd import ConstrainedDeformableRegistration
import numpy as np


def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(X[:, 0],  X[:, 1], X[:, 2], color='red', label='Target')
    ax.scatter(Y[:, 0],  Y[:, 1], Y[:, 2], color='blue', label='Source')
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
    src_id = np.int32([1,10,20,30, len(Y1)+1, len(Y1)+10, len(Y1)+20, len(Y1)+30])
    tgt_id = np.int32([1,10,20,30, len(X1)+1, len(X1)+10, len(X1)+20, len(X1)+30])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    callback = partial(visualize, ax=ax)

    reg = ConstrainedDeformableRegistration(**{'X': X, 'Y': Y}, e_alpha = 1e-8, source_id = src_id, target_id = tgt_id)
    reg.register(callback)
    plt.show()


if __name__ == '__main__':
    main()
