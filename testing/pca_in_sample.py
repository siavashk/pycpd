import json
import os

import numpy as np
import matplotlib.pyplot as plt
from examples.pca_skull_deformable_3D import plot_3d_interactive
from pycpd.pca_registration import PCADeformableRegistration
from pycpd.ssm import build_ssm

from pycpd.utility import get_slicer_positions_txt


def compute_rmse(A, B):
    """Compute Root Mean Square Error between two point clouds."""
    return np.sqrt(np.mean((A - B) ** 2))


def repeat_preserving_original(points, num_target_points):
    """Keep original points intact and fill extra with nearest repeats."""
    num_original = points.shape[0]

    if num_original == num_target_points:
        return points

    repeated_points = np.tile(points, (num_target_points // num_original + 1, 1))[:num_target_points]
    return repeated_points


def plot_point_sets(A, B, title="Point Cloud Comparison"):
    """Visualize two point clouds."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(A[:, 0], A[:, 1], A[:, 2], c='r', marker='o', label="Original (53 pts)")
    ax.scatter(B[:, 0], B[:, 1], B[:, 2], c='b', marker='^', label="Transformed (53 pts)")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.title(title)
    plt.show()


def PCA_rmse():
    skull_source = np.array([
        cp["position"]
        for cp in json.load(open("../data/mean/semilandmarks.json"))["markups"][0]["controlPoints"]
    ], dtype=float)

    skull_target = np.array([
        cp["position"]
        for cp in json.load(open("../data/semilandmarks/LG.ply_align.json"))["markups"][0]["controlPoints"]
    ], dtype=float)

    mean_test_source = np.array([
        cp['position']
        for cp in json.load(open("../data/mean/decaMeanModel.mrk.json"))["markups"][0]["controlPoints"]
    ])

    aligned_test_target = np.array([
        cp['position']
        for cp in json.load(open("../data/aligned_LMs/LG.ply_align.mrk.json"))["markups"][0]["controlPoints"]
    ])

    print(mean_test_source.shape)
    print(aligned_test_target.shape)

    X = skull_target  # Target (fixed landmarks)
    Y = skull_source



    plot_3d_interactive(skull_target, skull_source)

    # 1) Find all .json files in the directory
    json_dir = "../data/semilandmarks/"
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

    print(f"mean_test_source.shape: {mean_test_source.shape}")  # (53, 3)
    print(f"aligned_test_target.shape: {aligned_test_target.shape}")  # Should be (53, 3)
    print(f"X.shape: {X.shape}")  # Likely different
    print(f"Y.shape: {Y.shape}")  # Likely different
    print(f"U.shape: {U_reduced.shape}")  # Likely different

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
    mean_test_source_resampled = repeat_preserving_original(mean_test_source, 3872)
    # Step 2: Transform the resampled points using PCA model
    TY_resampled = reg.transform_point_cloud(Y=mean_test_source_resampled)
    # Step 3: Extract only the first 53 points from the transformed result
    TY_53 = TY_resampled[:53]

    print("resampled mean first 53 points",mean_test_source_resampled[:53])
    print("transformed first 53 points",TY_53)
    print("original 53 points",mean_test_source)

    rmse_before = compute_rmse(mean_test_source, aligned_test_target)
    print(f"RMSE before registration: {rmse_before:.6f}")
    # Step 4: Compute RMSE between the transformed 53 points and the original 53-point aligned_test_target
    rmse_53 = compute_rmse(TY_53, aligned_test_target)
    # rmse_53 should be less than rmse_before,
    print(f"RMSE between transformed 53 points and original 53 target points: {rmse_53:.6f}")

    # Plot original and transformed 53 points
    plot_point_sets(mean_test_source, aligned_test_target, title="Original vs Target 53 Points")
    plot_point_sets(TY_53, aligned_test_target, title="Transformed 53 Points vs Target")
    plot_point_sets(TY_53, mean_test_source_resampled[:53], title="Original vs Resampled 53 Points")


if __name__ == "__main__":
    PCA_rmse()