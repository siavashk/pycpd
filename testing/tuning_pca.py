import json
import os
import numpy as np
from itertools import product

from pycpd import DeformableRegistration
from pycpd.pca_registration import PCADeformableRegistration
from pycpd.ssm import build_ssm
from testing.pca_in_sample import compute_rmse, repeat_preserving_original

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

X = skull_target
Y = skull_source

# Define parameter ranges
alpha_values = [0.1]  # Deformation strength
w_values = [0.1]  # Weighting factor
max_iterations_values = [200]  # Iteration limits

# Store results
best_rmse = float("inf")
best_params = None
results = []

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

# Grid search loop
for alpha, w, max_iter in product(alpha_values, w_values, max_iterations_values):
    print(f"Testing alpha={alpha}, w={w}, max_iterations={max_iter}...")

    # Initialize PCA registration with different parameters
    reg_pca = PCADeformableRegistration(
        X=X,
        Y=Y,
        alpha=alpha,
        mean_shape=mean_shape,
        U=U_reduced,
        eigenvalues=eigenvalues,
        tolerance=0.0001,
        w=w,
        max_iterations=max_iter
    )

    reg_vanilla = DeformableRegistration(
        X=X,
        Y=Y,
    )

    # Register the full skull shape
    TY_pca, _ = reg_pca.register()

    TY_vanilla, _ = reg_vanilla.register()

    mean_test_source_resampled = repeat_preserving_original(mean_test_source, 3872)

    # Apply the same transformation to the 53 points
    TY_resampled_pca = reg_pca.transform_point_cloud(Y=mean_test_source_resampled)
    TY_resampled_vanilla = reg_vanilla.transform_point_cloud(Y=mean_test_source_resampled)

    TY_53_pca = TY_resampled_pca[:53]
    TY_53_vanilla = TY_resampled_vanilla[:53]



    # Compute RMSE for these settings
    rmse_mean_to_target = compute_rmse(mean_test_source, aligned_test_target)
    rmse_53_pca = compute_rmse(TY_53_pca, aligned_test_target)
    rmse_53_vanilla = compute_rmse(TY_53_vanilla, aligned_test_target)

    print("Mean to aligned target: ", rmse_mean_to_target)
    print("RMSE Vanilla Deformation: ", rmse_53_vanilla)
    print("RMSE PCA Deformation: ", rmse_53_pca)
    # Save the results
    results.append((alpha, w, max_iter, rmse_53_pca))

    # Track the best parameters
    if rmse_53_pca < best_rmse:
        best_rmse = rmse_53_pca
        best_params = (alpha, w, max_iter)

# Print the best found parameters
print("\nBest parameters found:")
print(f"alpha={best_params[0]}, w={best_params[1]}, max_iterations={best_params[2]}")
print(f"Best RMSE: {best_rmse:.6f}")
