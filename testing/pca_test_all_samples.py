import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pycpd.pca_registration import PCADeformableRegistration
from pycpd.deformable_registration import DeformableRegistration
from pycpd.ssm import build_ssm
from scipy.interpolate import Rbf
import pandas as pd
from pathlib import Path


def compute_rmse(A, B):
    """Compute Root Mean Square Error between two point clouds."""
    return np.sqrt(np.mean((A - B) ** 2))


def plot_point_sets(A, B, title="Point Cloud Comparison", A_label="Original", B_label="Target", save_dir=None):
    """Visualize two point clouds."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(A[:, 0], A[:, 1], A[:, 2], c='r', marker='o', label=A_label)
    ax.scatter(B[:, 0], B[:, 1], B[:, 2], c='b', marker='^', label=B_label)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.title(title)

    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{title.replace(' ', '_')}.png"))
    else:
        plt.savefig(f"{title.replace(' ', '_')}.png")

    plt.close(fig)  # Close to avoid showing plots when running in batch mode


def calculate_tps_transform(source_points, target_points):
    """Calculate a Thin Plate Spline transform between source and target points."""
    # Create separate RBF interpolators for each coordinate (x, y, z)
    rbf_x = Rbf(source_points[:, 0], source_points[:, 1], source_points[:, 2], target_points[:, 0],
                function='thin_plate')
    rbf_y = Rbf(source_points[:, 0], source_points[:, 1], source_points[:, 2], target_points[:, 1],
                function='thin_plate')
    rbf_z = Rbf(source_points[:, 0], source_points[:, 1], source_points[:, 2], target_points[:, 2],
                function='thin_plate')

    def transform_function(points):
        x_transformed = rbf_x(points[:, 0], points[:, 1], points[:, 2])
        y_transformed = rbf_y(points[:, 0], points[:, 1], points[:, 2])
        z_transformed = rbf_z(points[:, 0], points[:, 1], points[:, 2])
        return np.vstack([x_transformed, y_transformed, z_transformed]).T

    return transform_function


def repeat_preserving_original(points, num_target_points):
    """Keep original points intact and fill extra with nearest repeats."""
    num_original = points.shape[0]

    if num_original == num_target_points:
        return points

    repeated_points = np.tile(points, (num_target_points // num_original + 1, 1))[:num_target_points]
    return repeated_points


def create_results_visualization(specimen_name, deca_mean_source, aligned_test_target, deca_mean_transformed_pca,
                                 deca_mean_transformed_traditional, pca_rmse, traditional_rmse, save_dir):
    """Create a detailed visualization comparing all methods for a specimen."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Original points
    ax.scatter(deca_mean_source[:, 0], deca_mean_source[:, 1], deca_mean_source[:, 2],
               c='k', marker='o', s=30, label="Original DECA Mean")

    # Target points
    ax.scatter(aligned_test_target[:, 0], aligned_test_target[:, 1], aligned_test_target[:, 2],
               c='g', marker='*', s=50, label=f"{specimen_name} Target")

    # PCA transformed
    ax.scatter(deca_mean_transformed_pca[:, 0], deca_mean_transformed_pca[:, 1], deca_mean_transformed_pca[:, 2],
               c='r', marker='^', s=30, label=f"PCA-CPD (RMSE: {pca_rmse:.4f})")

    # Traditional transformed
    ax.scatter(deca_mean_transformed_traditional[:, 0], deca_mean_transformed_traditional[:, 1],
               deca_mean_transformed_traditional[:, 2],
               c='b', marker='s', s=30, label=f"Traditional CPD (RMSE: {traditional_rmse:.4f})")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.title(f"Registration Methods Comparison - {specimen_name}")
    plt.savefig(os.path.join(save_dir, f"{specimen_name}_Comparison.png"))
    plt.close(fig)


def process_specimen(specimen_file, skull_source, mean_shape, U_reduced, eigenvalues, deca_mean_source, results_dir):
    """Process a single specimen file through both registration methods."""
    specimen_name = os.path.basename(specimen_file).split('.')[0]
    print(f"\n===== Processing {specimen_name} =====")

    # Create a specimen-specific output directory
    specimen_dir = os.path.join(results_dir, specimen_name)
    os.makedirs(specimen_dir, exist_ok=True)

    # Load target landmarks
    try:
        aligned_test_target = np.array([
            cp['position']
            for cp in json.load(open(specimen_file))["markups"][0]["controlPoints"]
        ])
    except Exception as e:
        print(f"Error loading {specimen_file}: {e}")
        return None

    # Load target semilandmarks
    try:
        skull_target_file = f"../data/semilandmarks/{specimen_name}.ply_align.json"
        skull_target = np.array([
            cp["position"]
            for cp in json.load(open(skull_target_file))["markups"][0]["controlPoints"]
        ], dtype=float)
    except Exception as e:
        print(f"Error loading semilandmarks for {specimen_name}: {e}")
        return None

    # Calculate initial RMSE
    initial_rmse = compute_rmse(deca_mean_source, aligned_test_target)
    print(f"Initial RMSE between DECA mean and {specimen_name} target: {initial_rmse:.6f}")

    # Plot original data
    plot_point_sets(deca_mean_source, aligned_test_target,
                    title=f"Original: DECA Mean vs {specimen_name} Target",
                    A_label="DECA Mean (53 pts)",
                    B_label=f"{specimen_name} Target (53 pts)",
                    save_dir=specimen_dir)

    # Run PCA-based CPD
    print(f"Running PCA-based CPD for {specimen_name}...")
    pca_reg = PCADeformableRegistration(
        X=skull_target,  # Target (fixed)
        Y=skull_source,  # Source (moving)
        alpha=0.1,  # PCA parameter
        mean_shape=mean_shape,
        U=U_reduced,
        eigenvalues=eigenvalues,
        tolerance=0.001,
        w=0.1,  # EM parameter
        max_iterations=150
    )

    pca_transformed, _ = pca_reg.register()

    # Step 2: Calculate TPS transform for PCA result
    print(f"Calculating TPS transform from PCA registration for {specimen_name}...")
    pca_tps_transform = calculate_tps_transform(skull_source, pca_transformed)

    # Step 3: Apply TPS transform to 53-point DECA mean
    deca_mean_transformed_pca = pca_tps_transform(deca_mean_source)

    # Calculate RMSE after PCA-based transformation
    pca_rmse = compute_rmse(deca_mean_transformed_pca, aligned_test_target)
    print(f"RMSE after PCA-based transformation (TPS) for {specimen_name}: {pca_rmse:.6f}")

    # Plot transformed vs target for PCA method
    plot_point_sets(deca_mean_transformed_pca, aligned_test_target,
                    title=f"PCA-CPD: Transformed DECA Mean vs {specimen_name}",
                    A_label="Transformed DECA Mean (PCA)",
                    B_label=f"{specimen_name} Target",
                    save_dir=specimen_dir)

    # Step 4: Run traditional CPD
    print(f"Running Traditional CPD for {specimen_name}...")
    traditional_reg = DeformableRegistration(
        X=skull_target,  # Target (fixed)
        Y=skull_source,  # Source (moving)
    )

    traditional_transformed, _ = traditional_reg.register()

    # Use transform_point_cloud for traditional CPD
    print(f"Using transform_point_cloud for traditional CPD on {specimen_name}...")

    # Repeat the 53 points to match source points
    num_points_in_source = skull_source.shape[0]
    print(f"Repeating 53 points to match {num_points_in_source} points...")
    deca_mean_source_resampled = repeat_preserving_original(deca_mean_source, num_points_in_source)

    # Apply transformation using transform_point_cloud
    TY_resampled_vanilla = traditional_reg.transform_point_cloud(Y=deca_mean_source_resampled)

    # Extract first 53 points
    deca_mean_transformed_traditional = TY_resampled_vanilla[:53]

    print(f"Original 53 points shape for {specimen_name}: {deca_mean_source.shape}")

    # Calculate RMSE after traditional CPD
    traditional_rmse = compute_rmse(deca_mean_transformed_traditional, aligned_test_target)
    print(f"RMSE after traditional CPD (direct transform) for {specimen_name}: {traditional_rmse:.6f}")

    # Plot transformed vs target for traditional method
    plot_point_sets(deca_mean_transformed_traditional, aligned_test_target,
                    title=f"Traditional CPD: Transformed DECA Mean vs {specimen_name}",
                    A_label="Transformed DECA Mean (Traditional)",
                    B_label=f"{specimen_name} Target",
                    save_dir=specimen_dir)

    # Compare PCA vs Traditional methods
    plot_point_sets(deca_mean_transformed_pca, deca_mean_transformed_traditional,
                    title=f"PCA vs Traditional CPD - {specimen_name}",
                    A_label="PCA-CPD Transformed",
                    B_label="Traditional CPD Transformed",
                    save_dir=specimen_dir)

    # Calculate improvement metrics
    pca_improvement = ((initial_rmse - pca_rmse) / initial_rmse) * 100
    traditional_improvement = ((initial_rmse - traditional_rmse) / initial_rmse) * 100

    method_difference = ((traditional_rmse - pca_rmse) / traditional_rmse) * 100 if traditional_rmse > pca_rmse else (
                                                                                                                             (
                                                                                                                                         pca_rmse - traditional_rmse) / pca_rmse) * 100

    better_method = 'PCA' if pca_rmse < traditional_rmse else 'Traditional'

    # Print summary for this specimen
    print(f"\n--- Summary for {specimen_name} ---")
    print(f"Initial RMSE: {initial_rmse:.6f}")
    print(f"PCA-CPD RMSE: {pca_rmse:.6f} (Improvement: {pca_improvement:.2f}%)")
    print(f"Traditional CPD RMSE: {traditional_rmse:.6f} (Improvement: {traditional_improvement:.2f}%)")
    print(f"Difference between methods: {abs(method_difference):.2f}% ({better_method} better)")

    # Create the detailed comparison visualization
    create_results_visualization(
        specimen_name,
        deca_mean_source,
        aligned_test_target,
        deca_mean_transformed_pca,
        deca_mean_transformed_traditional,
        pca_rmse,
        traditional_rmse,
        specimen_dir
    )

    # Create a bar chart for RMSE comparison
    plt.figure(figsize=(10, 6))
    methods = ['Initial', 'PCA-CPD', 'Traditional CPD']
    rmse_values = [initial_rmse, pca_rmse, traditional_rmse]
    colors = ['gray', 'red', 'blue']

    plt.bar(methods, rmse_values, color=colors)
    plt.ylabel('RMSE')
    plt.title(f'RMSE Comparison - {specimen_name}')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add RMSE values on top of bars
    for i, v in enumerate(rmse_values):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')

    plt.savefig(os.path.join(specimen_dir, f"{specimen_name}_RMSE_Comparison.png"))
    plt.close()

    # Return results for this specimen
    return {
        'specimen': specimen_name,
        'initial_rmse': initial_rmse,
        'pca_rmse': pca_rmse,
        'traditional_rmse': traditional_rmse,
        'pca_improvement': pca_improvement,
        'traditional_improvement': traditional_improvement,
        'method_difference': abs(method_difference),
        'better_method': better_method
    }


def main():
    # Create results directory
    results_dir = "batch_comparison_results"
    os.makedirs(results_dir, exist_ok=True)

    # Load the DECA mean source (same for all specimens)
    deca_mean_source = np.array([
        cp['position']
        for cp in json.load(open("../data/mean/decaMeanModel.mrk.json"))["markups"][0]["controlPoints"]
    ])

    # Load the skull source (mean semilandmarks)
    skull_source = np.array([
        cp["position"]
        for cp in json.load(open("../data/mean/semilandmarks.json"))["markups"][0]["controlPoints"]
    ], dtype=float)

    # Build the Statistical Shape Model (once for all specimens)
    print("Building Statistical Shape Model...")
    json_dir = "../data/semilandmarks/"
    json_files = [f for f in os.listdir(json_dir) if f.lower().endswith(".json")]

    all_shapes = []
    for fname in json_files:
        path = os.path.join(json_dir, fname)
        with open(path, "r") as f:
            data = json.load(f)
        cpoints = data.get("markups", [])[0].get("controlPoints", [])
        arr = np.array([cp["position"] for cp in cpoints], dtype=float)
        all_shapes.append(arr)

    shapes_np = np.stack(all_shapes, axis=0)
    mean_shape, U_reduced, eigenvalues, num_modes = build_ssm(shapes_np, variance_threshold=0.95)
    print(f"Number of shape modes retained: {num_modes}")

    # Find all aligned landmark files
    aligned_dir = "../data/aligned_LMs/"
    aligned_files = [os.path.join(aligned_dir, f) for f in os.listdir(aligned_dir) if f.endswith(".mrk.json")]

    if not aligned_files:
        print(f"No .mrk.json files found in {aligned_dir}")
        return

    print(f"Found {len(aligned_files)} specimen files to process")

    # Process each specimen
    all_results = []
    for specimen_file in aligned_files:
        result = process_specimen(
            specimen_file,
            skull_source,
            mean_shape,
            U_reduced,
            eigenvalues,
            deca_mean_source,
            results_dir
        )

        if result:
            all_results.append(result)

    # Create a summary dataframe
    if all_results:
        df = pd.DataFrame(all_results)

        # Save to CSV
        df.to_csv(os.path.join(results_dir, "batch_comparison_summary.csv"), index=False)

        # Print summary statistics
        print("\n===== OVERALL RESULTS =====")
        print(f"Total specimens processed: {len(all_results)}")
        print(f"Average Initial RMSE: {df['initial_rmse'].mean():.6f}")
        print(f"Average PCA-CPD RMSE: {df['pca_rmse'].mean():.6f}")
        print(f"Average Traditional CPD RMSE: {df['traditional_rmse'].mean():.6f}")
        print(f"Average PCA Improvement: {df['pca_improvement'].mean():.2f}%")
        print(f"Average Traditional Improvement: {df['traditional_improvement'].mean():.2f}%")

        # Count which method performed better
        method_counts = df['better_method'].value_counts()
        for method, count in method_counts.items():
            print(f"{method} was better in {count} cases ({count / len(df) * 100:.2f}%)")

        # Create overall comparison chart
        plt.figure(figsize=(12, 8))

        # Sort by initial RMSE for better visualization
        df_sorted = df.sort_values('initial_rmse')

        x = range(len(df_sorted))
        width = 0.25

        plt.bar([i - width for i in x], df_sorted['initial_rmse'], width, label='Initial RMSE', color='gray')
        plt.bar([i for i in x], df_sorted['pca_rmse'], width, label='PCA-CPD RMSE', color='red')
        plt.bar([i + width for i in x], df_sorted['traditional_rmse'], width, label='Traditional CPD RMSE',
                color='blue')

        plt.xlabel('Specimen')
        plt.ylabel('RMSE')
        plt.title('RMSE Comparison Across All Specimens')
        plt.xticks([i for i in x], df_sorted['specimen'], rotation=90)
        plt.legend()
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(results_dir, "Overall_RMSE_Comparison.png"))

        # Create improvement comparison
        plt.figure(figsize=(12, 8))

        # Sort by PCA improvement for better visualization
        df_sorted = df.sort_values('pca_improvement')

        x = range(len(df_sorted))

        plt.bar([i - width / 2 for i in x], df_sorted['pca_improvement'], width, label='PCA-CPD Improvement',
                color='red')
        plt.bar([i + width / 2 for i in x], df_sorted['traditional_improvement'], width,
                label='Traditional CPD Improvement', color='blue')

        plt.xlabel('Specimen')
        plt.ylabel('Improvement (%)')
        plt.title('Registration Improvement Across All Specimens')
        plt.xticks([i for i in x], df_sorted['specimen'], rotation=90)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(results_dir, "Overall_Improvement_Comparison.png"))

        print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()