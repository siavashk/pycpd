from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pycpd import deformable_registration
import numpy as np
import time

def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(X[:,0],  X[:,1], X[:,2], color='red', label='Target')
    ax.scatter(Y[:,0],  Y[:,1], Y[:,2], color='blue', label='Source')
    ax.text2D(0.87, 0.92, 'Iteration: {:d}\nError: {:06.4f}'.format(iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)

def main():
    fish_target = np.loadtxt('data/fish_target.txt')
    X1 = np.zeros((fish_target.shape[0], fish_target.shape[1] + 1))
    X1[:,:-1] = fish_target

    theta = np.pi / 6.0
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    t = np.array([0.5, 1.0])

    X1[:,:-1] = np.dot(X1[:,:-1], R) + np.tile(t, (np.shape(X1[:,:-1])[0], 1))
    # X2 = np.ones((fish_target.shape[0], fish_target.shape[1] + 1))
    # X2[:,:-1] = fish_target
    # X = np.vstack((X1, X2))

    fish_source = np.loadtxt('data/fish_source.txt')
    Y1 = np.zeros((fish_source.shape[0], fish_source.shape[1] + 1))
    Y1[:,:-1] = fish_source
    # Y2 = np.ones((fish_source.shape[0], fish_source.shape[1] + 1))
    # Y2[:,:-1] = fish_source
    # Y = np.vstack((Y1, Y2))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    callback = partial(visualize, ax=ax)

    reg = deformable_registration(**{ 'X': X1, 'Y': Y1 })
    reg.register(callback)
    plt.show()

if __name__ == '__main__':
    main()
