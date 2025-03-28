import numpy as np
import json
import os
from matplotlib import pyplot as plt


def is_positive_semi_definite(R):
    if not isinstance(R, (np.ndarray, np.generic)):
        raise ValueError('Encountered an error while checking if the matrix is positive semi definite. \
            Expected a numpy array, instead got : {}'.format(R))
    return np.all(np.linalg.eigvals(R) > 0)

def gaussian_kernel(X, beta, Y=None):
    if Y is None:
        Y = X
    diff = X[:, None, :] - Y[None, :,  :]
    diff = np.square(diff)
    diff = np.sum(diff, 2)
    return np.exp(-diff / (2 * beta**2))

def pca_kernel(X, mean_shape, U, eigenvalues):
    """
    Compute a PCA-based kernel matrix for shape deformation.
    """
    n_points_x = X.shape[0]

    # Ensure U is properly shaped
    if len(U.shape) == 3:
        U_resh = U  # Already correctly shaped
    else:
        num_modes = U.size // (n_points_x * 3)  # Compute dynamically
        if num_modes * n_points_x * 3 != U.size:
            raise ValueError(f"U cannot be reshaped correctly. U.shape: {U.shape}, Expected: ({n_points_x}, 3, ?)")
        U_resh = U.reshape(n_points_x, 3, num_modes)

    print(f"Reshaped U: {U_resh.shape}")

    # Mode weights computation
    M = np.einsum('ikm,jkm->ijm', U_resh, U_resh)

    # Ensure eigenvalues are the correct length
    if eigenvalues.shape[0] > M.shape[-1]:
        eigenvalues = eigenvalues[:M.shape[-1]]  # Slice down if too many modes

    print(f"eigenvalues.shape: {eigenvalues.shape}, expected: ({M.shape[-1]},)")

    # Normalize using eigenvalues
    M /= (np.log(eigenvalues + 1e-8) + 1)

    # Compute final kernel matrix
    K = np.sum(M, axis=2)

    # Normalize to [0,1]
    K = K / np.max(K)

    return K

def low_rank_eigen(G, num_eig):
    """
    Calculate num_eig eigenvectors and eigenvalues of gaussian matrix G.
    Enables lower dimensional solving.
    """
    S, Q = np.linalg.eigh(G)
    eig_indices = list(np.argsort(np.abs(S))[::-1][:num_eig])
    Q = Q[:, eig_indices]  # eigenvectors
    S = S[eig_indices]  # eigenvalues.
    return Q, S


def get_slicer_positions_txt(json_file_path):
    """
    Reads a Slicer Markups JSON file from a given path, extracts position data,
    and returns it as a formatted string in scientific notation.

    :param json_file_path: Path to the JSON file.
    :return: A formatted string of positions or an error message if the file is missing.
    """
    if not os.path.exists(json_file_path):
        return f"Error: File not found at {json_file_path}"

    # Read and parse the JSON file
    with open(json_file_path, "r", encoding="utf-8") as file:
        json_data = json.load(file)

    # Extract control points from the first markup entry
    control_points = json_data.get("markups", [])[0].get("controlPoints", [])

    # Format positions
    formatted_positions = "\n".join(
        " ".join(f"{coord:.18e}" for coord in entry["position"]) for entry in control_points
    )

    return formatted_positions

def calculate_registration_metrics(source, target):
    """Calculate alignment metrics between source and target point clouds."""
    diff = source - target
    return {
        'rmse': np.sqrt(np.mean(np.sum(diff**2, axis=1))),
        'mae': np.mean(np.abs(diff)),
        'max_error': np.max(np.abs(diff)),
        'rmse_per_axis': np.sqrt(np.mean(diff**2, axis=0))
    }


def plot_metrics_comparison(results):
    """Plots comparison of RMSE and execution time given registration results."""
    # Extract registration method names, rmse values and execution times
    method_names = list(results.keys())
    rmse_values = [results[m]['metrics']['rmse'] for m in method_names]
    times = [results[m]['time'] for m in method_names]

    # Bar chart for RMSE
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.bar(method_names, rmse_values, color='skyblue')
    plt.title("RMSE Comparison")
    plt.ylabel("RMSE")
    plt.xlabel("Method")

    # Bar chart for Execution Time
    plt.subplot(1, 2, 2)
    plt.bar(method_names, times, color='salmon')
    plt.title("Execution Time")
    plt.ylabel("Time (seconds)")
    plt.xlabel("Method")

    plt.tight_layout()
    plt.show()


def plot_registration_comparison(X, Y, results):
    """Show side-by-side plots comparing the point set registration for each method."""

    dim = X.shape[1]
    method_names = list(results.keys())

    num_methods = len(method_names)
    plt.figure(figsize=(6 * num_methods, 5))

    for i, m in enumerate(method_names, 1):
        TY = results[m]['transformed_points']

        plt.subplot(1, num_methods, i)
        if dim == 2:
            # 2D scatter
            plt.scatter(X[:, 0], X[:, 1], c='red', label='Target (X)', s=8)
            plt.scatter(TY[:, 0], TY[:, 1], c='blue', label=f'{m} Registered (Y)', s=8)
            plt.scatter(Y[:, 0], Y[:, 1], c='green', label='Original Y', s=8, alpha=0.3)
            plt.title(f"{m} Registration")
            plt.legend(loc='upper right')
        elif dim == 3:
            # 3D scatter example
            ax = plt.gca(projection='3d')
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='red', label='Target (X)', s=8)
            ax.scatter(TY[:, 0], TY[:, 1], TY[:, 2], c='blue', label=f'{m} Registered (Y)', s=8)
            ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c='green', label='Original Y', s=8, alpha=0.3)
            ax.set_title(f"{m} Registration")
            ax.legend()
        else:
            raise ValueError("dim must be 2 or 3.")

    plt.tight_layout()
    plt.show()





