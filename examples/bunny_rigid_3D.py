import argparse
from functools import partial
import matplotlib.pyplot as plt
from pycpd import RigidRegistration
import numpy as np
import os


def visualize(iteration, error, X, Y, ax, fig, save_fig=False):
    plt.cla()
    ax.scatter(X[:, 0],  X[:, 1], X[:, 2], color='red', label='Target')
    ax.scatter(Y[:, 0],  Y[:, 1], Y[:, 2], color='blue', label='Source')
    ax.text2D(0.87, 0.92, 'Iteration: {:d}\nQ: {:06.4f}'.format(
        iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    ax.view_init(90, -90)
    if save_fig is True:
        ax.set_axis_off()

    plt.draw()
    if save_fig is True:
        os.makedirs("./images/rigid_bunny/", exist_ok=True)
        fig.savefig("./images/rigid_bunny/rigid_bunny_3D_{:04}.tiff".format(iteration), dpi=600)  # Used for making gif.
    plt.pause(0.001)


def main(save=False):
    print(save)
    X = np.loadtxt('data/bunny_target.txt')
    # synthetic data, equaivalent to X + 1
    Y = np.loadtxt('data/bunny_source.txt')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    callback = partial(visualize, ax=ax, fig=fig, save_fig=save[0] if type(save) is list else save)

    reg = RigidRegistration(**{'X': X, 'Y': Y})
    reg.register(callback)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rigid registration example")
    parser.add_argument(
        "-s",
        "--save",
        type=bool,
        nargs="+",
        default=False,
        help="True or False - to save figures of the example for a GIF etc.",
    )
    args = parser.parse_args()
    print(args)

    main(**vars(args))
