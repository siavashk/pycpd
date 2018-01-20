from functools import partial
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pycpd import affine_registration
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

    X1 = np.zeros((fish['X'].shape[0], fish['X'].shape[1] + 1))
    X1[:,:-1] = fish['X']
    X2 = np.ones((fish['X'].shape[0], fish['X'].shape[1] + 1))
    X2[:,:-1] = fish['X']
    X = np.vstack((X1, X2))

    Y1 = np.zeros((fish['Y'].shape[0], fish['Y'].shape[1] + 1))
    Y1[:,:-1] = fish['Y']
    Y2 = np.ones((fish['Y'].shape[0], fish['Y'].shape[1] + 1))
    Y2[:,:-1] = fish['Y']
    Y = np.vstack((Y1, Y2))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    callback = partial(visualize, ax=ax)

    reg = affine_registration(X, Y)
    reg.register(callback)
    plt.show()

if __name__ == '__main__':
    main()
