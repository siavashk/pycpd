import argparse
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import os
from pycpd.pca_registration import PCADeformableRegistration
from pycpd.ssm import build_ssm
import json
from pycpd.utility import get_slicer_positions_txt
import plotly.graph_objects as go

def plot_3d_interactive(X, Y):
    """
    X and Y are (n_points, 3).
    We'll create two separate traces so we can toggle them on/off in the legend.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=X[:,0], y=X[:,1], z=X[:,2],
        mode='markers',
        marker=dict(size=3, color='red'),
        name='Target (X)'
    ))
    fig.add_trace(go.Scatter3d(
        x=Y[:,0], y=Y[:,1], z=Y[:,2],
        mode='markers',
        marker=dict(size=3, color='blue'),
        name='Source (Y)'
    ))

    # Rotate, zoom, or hide data sets by clicking on legend entries in the rendered figure
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title="Interactive 3D Plot (Plotly)"
    )
    fig.show()

# Usage:
# plot_3d_interactive(skull_target, skull_source)


def main(save=False):
    plt.close('all')  # at the start of main()
    print(f"Save figures: {save}")

    # Source
    skull_source = np.array([
        cp["position"]
        for cp in json.load(open("./data/mean/semilandmarks.json"))["markups"][0]["controlPoints"]
    ], dtype=float)

    # Target
    skull_target = np.array([
        cp["position"]
        for cp in json.load(open("./data/semilandmarks/LG.ply_align.json"))["markups"][0]["controlPoints"]
    ], dtype=float)

    X = skull_target  # Target (fixed landmarks)
    Y = skull_source

    # Plot before registration
    plot_3d_interactive(skull_target, skull_source)


    # 1) Find all .json files in the directory
    json_dir = "./data/semilandmarks/"
    files_in_dir = os.listdir(json_dir)
    json_files = [f for f in files_in_dir if f.lower().endswith(".json")]

    if not json_files:
        raise ValueError(f"No JSON files found in directory: {json_dir}")

    all_shapes = []
    for fname in json_files:
        path = os.path.join(json_dir, fname)
        with open(path, "r") as f:
            data = json.load(f)
        # Extract positions
        cpoints = data.get("markups", [])[0].get("controlPoints", [])
        arr = np.array([cp["position"] for cp in cpoints], dtype=float)
        # e.g. arr has shape (n_points, D)

        all_shapes.append(arr)

    # Make sure they all share the same shape, e.g. (n_points, D)
    ref_shape = all_shapes[0].shape
    for i, shape_ in enumerate(all_shapes):
        if shape_.shape != ref_shape:
            raise ValueError(f"Mismatch in shape at index {i}: got {shape_.shape}, expected {ref_shape}.")

    # Stack them => (n_shapes, n_points, D)
    shapes_np = np.stack(all_shapes, axis=0)
    print("shapes_np shape:", shapes_np.shape)


    # Build the Statistical Shape Model
    mean_shape, U_reduced, eigenvalues, num_modes = build_ssm(shapes_np, variance_threshold=0.95)
    print(f"Number of shape modes retained: {num_modes}")

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

    TY, _ = reg.register()

    # 5. Plot after registration
    plot_3d_interactive(skull_target, TY)
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