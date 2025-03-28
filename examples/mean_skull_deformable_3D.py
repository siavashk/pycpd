from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pycpd import ConstrainedDeformableRegistration
import numpy as np

from pycpd.utility import get_slicer_positions_txt

txt_file_source = get_slicer_positions_txt('data/mean/semilandmarks.json')
txt_file_target = get_slicer_positions_txt('data/semilandmarks/B6AF1_J.ply_align.json')
fish_source = np.fromstring(txt_file_source, sep=' ').reshape(-1, 3)
fish_target = np.fromstring(txt_file_target, sep=' ').reshape(-1, 3)
marker_size = 100
N_pts_include = 3872
IDs = [1,10,20,30]

IDs_Y = [i for i in IDs if i < 3872]  # Avoid exceeding dataset size
IDs_X = [i for i in IDs if i < 3872]

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
    skull_source = (np.fromstring(get_slicer_positions_txt('data/mean/semilandmarks.json'), sep=' ')
                   .reshape(-1, 3))
    skull_target = (np.fromstring(get_slicer_positions_txt('data/semilandmarks/B6AF1_J.ply_align.json'), sep=' ')
                   .reshape(-1, 3))

    print(f"Shape of fish_target (X): {fish_target.shape}")  # Expected: (3872, 3)
    print(f"Shape of fish_source (Y): {fish_source.shape}")





    # print(Y1.shape)
    # print(X1.shape)

    X = fish_target  # Target (fixed landmarks)
    Y = fish_source


    # select fixed correspondences
    src_id = np.array(IDs_Y, dtype=np.int32)
    tgt_id = np.array(IDs_X, dtype=np.int32)

    print("Max src_id:", np.max(src_id))
    print("Max tgt_id:", np.max(tgt_id))
    print("Valid range: 0 -", X.shape[0] - 1)

    if np.max(src_id) >= Y.shape[0] or np.max(tgt_id) >= X.shape[0]:
        raise ValueError("Error: source_id or target_id index is out of range!")

    print("Shape of X:", X.shape)
    print("Shape of Y:", Y.shape)
    print("Max source_id:", np.max(src_id))
    print("Max target_id:", np.max(tgt_id))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    callback = partial(visualize, ax=ax)

    reg = ConstrainedDeformableRegistration(**{'X': X, 'Y': Y}, e_alpha = 1e-8, source_id = src_id, target_id = tgt_id)
    reg.register(callback)
    print("Finished!")
    plt.show()


if __name__ == '__main__':
    main()
