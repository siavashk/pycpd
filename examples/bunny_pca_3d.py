import argparse
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import os
from pycpd.pca_registration import PCADeformableRegistration
from pycpd.ssm import build_ssm


def visualize(iteration, error, X, Y, ax, fig, save_fig=False):
    plt.cla()
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], color='red', label='Target')
    ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], color='blue', label='Source')
    ax.text2D(0.87, 0.92, 'Iteration: {:d}\nQ: {:06.4f}'.format(
        iteration, error), horizontalalignment='center', verticalalignment='center',
              transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    ax.view_init(90, -90)
    if save_fig:
        ax.set_axis_off()

    plt.draw()
    if save_fig:
        os.makedirs("./images/pca_bunny/", exist_ok=True)
        fig.savefig("./images/pca_bunny/pca_bunny_3D_{:04}.tiff".format(iteration),
                    dpi=600)
    plt.pause(0.001)


def main(save=False):
    print(f"Save figures: {save}")

    # Load the bunny data
    X = np.loadtxt('data/bunny_target.txt')
    Y = np.loadtxt('data/bunny_source.txt')

    # Create synthetic training shapes for SSM
    num_training_shapes = 20
    training_shapes = []
    np.random.seed(42)

    for i in range(num_training_shapes):
        # Create synthetic deformation with varying parameters
        noise_scale = np.random.uniform(0.02, 0.08)
        noise = np.random.normal(0, noise_scale, X.shape)
        rotation = np.random.uniform(-0.2, 0.2)
        scale = np.random.uniform(0.95, 1.05)

        # Apply rotation
        rot_matrix = np.array([
            [np.cos(rotation), -np.sin(rotation), 0],
            [np.sin(rotation), np.cos(rotation), 0],
            [0, 0, 1]
        ])

        # Apply transformations
        deformed_shape = scale * np.dot(X + noise, rot_matrix)
        training_shapes.append(deformed_shape)

    training_shapes = np.array(training_shapes)

    # Build the Statistical Shape Model
    mean_shape, U_reduced, eigenvalues, num_modes = build_ssm(training_shapes, variance_threshold=0.98)
    print(f"Number of shape modes retained: {num_modes}")

    # Setup visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    callback = partial(visualize, ax=ax, fig=fig,
                       save_fig=save[0] if isinstance(save, list) else save)

    # Initialize and run the PCA-based registration with better parameters
    reg = PCADeformableRegistration(
        X=X,
        Y=Y,
        alpha=0.1,  # PCA parameter
        mean_shape=mean_shape,  # PCA parameter
        U=U_reduced,  # PCA parameter
        eigenvalues=eigenvalues,  # PCA parameter
        tolerance=0.001,  # Increased tolerance
        w=0.1,  # EM parameter
        max_iterations=150  # More iterations allowed
    )

    reg.register(callback)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PCA-based registration example")
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