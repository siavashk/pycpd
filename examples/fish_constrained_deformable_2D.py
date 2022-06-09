from functools import partial
import matplotlib.pyplot as plt
from pycpd import ConstrainedDeformableRegistration
import numpy as np
import time


def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(X[:, 0],  X[:, 1], color='red', label='Target')
    ax.scatter(Y[:, 0],  Y[:, 1], color='blue', label='Source')
    plt.text(0.87, 0.92, 'Iteration: {:d}'.format(
        iteration), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)


def main():
    X = np.loadtxt('data/fish_target.txt')
    Y = np.loadtxt('data/fish_source.txt')

    # simulate a pointcloud missing certain parts
    X = X[:61]
    # select fixed correspondences
    src_id = np.int32([1,10,20,30])
    tgt_id = np.int32([1,10,20,30])

    fig = plt.figure()
    fig.add_axes([0, 0, 1, 1])
    callback = partial(visualize, ax=fig.axes[0])

    # e_alpha can be tuned (default: 1e-8). the smaller the value, the more confidence it will have in the correspodence
    reg = ConstrainedDeformableRegistration(**{'X': X, 'Y': Y}, e_alpha = 1e-8, source_id = src_id, target_id = tgt_id)  
    reg.register(callback)
    plt.show()


if __name__ == '__main__':
    main()
