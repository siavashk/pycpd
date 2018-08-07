from functools import partial
from scipy.io import loadmat
import matplotlib.pyplot as plt
from pycpd import jrmpc_rigid
import numpy as np
import time

def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(X[:,0] ,  X[:,1], color='red', label='Target')
    ax.scatter(Y[:,0] ,  Y[:,1], color='blue', label='Source')
    plt.text(0.87, 0.92, 'Iteration: {:d}\nError: {:06.4f}'.format(iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)

def main():
    fish = loadmat('../data/fish.mat')
    X = fish['X']
    Y = fish['Y']

    fig = plt.figure()
    fig.add_axes([0, 0, 1, 1])
    callback = partial(visualize, ax=fig.axes[0])

    reg = jrmpc_rigid([X, Y])
    reg.print_self()

if __name__ == '__main__':
    main()
