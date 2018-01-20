from functools import partial
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pycpd import deformable_registration
import numpy as np
import time

def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(X[:,0],  X[:,1], X[:,2], color='red')
    ax.scatter(Y[:,0],  Y[:,1], Y[:,2], color='blue')
    plt.draw()
    print("iteration %d, error %.5f" % (iteration, error))
    plt.pause(0.001)

def main():
    fish = loadmat('../data/fish.mat')
    X = np.zeros((fish['X'].shape[0], fish['X'].shape[1] + 1))
    X[:,:-1] = fish['X']

    Y = np.zeros((fish['Y'].shape[0], fish['Y'].shape[1] + 1))
    Y[:,:-1] = fish['Y']

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    callback = partial(visualize, ax=ax)

    reg = deformable_registration(X, Y)
    reg.register(callback)
    plt.show()

if __name__ == '__main__':
    main()
