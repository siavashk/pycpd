from functools import partial
from scipy.io import loadmat
import matplotlib.pyplot as plt
from pycpd import RigidRegistration
import numpy as np
import time

def visualize(X, Y, ax):
    plt.cla()
    ax.scatter(X[:,0] ,  X[:,1], color='red')
    ax.scatter(Y[:,0] ,  Y[:,1], color='blue')
    plt.draw()
    plt.pause(0.001)

def main():
    fish = loadmat('./data/fish.mat')
    X = fish['X']
    Y = fish['Y']

    fig = plt.figure()
    fig.add_axes([0, 0, 1, 1])
    callback = partial(visualize, ax=fig.axes[0])

    reg = RigidRegistration(X, Y)
    reg.register(callback)
    plt.show()

if __name__ == '__main__':
    main()
