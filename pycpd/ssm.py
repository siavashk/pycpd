import numpy as np
from sklearn.decomposition import PCA

# build ssm
def build_ssm(shapes, variance_threshold=0.95):
    """
    Builds a Statistical Shape Model (SSM) using PCA and retains dominant modes.
    """
    num_shapes, num_points, dim = shapes.shape

    # Flatten shapes into vectors
    flattened_shapes = shapes.reshape(num_shapes, -1)  # Shape: (n_shapes, n_points * dim)

    # Compute Mean Shape
    mean_shape = np.mean(flattened_shapes, axis=0)  # Shape: (n_points * dim,)
    centered_shapes = flattened_shapes - mean_shape

    # Perform PCA
    U, s, _ = np.linalg.svd(centered_shapes, full_matrices=False)

    # Convert singular values to eigenvalues
    eigenvalues = (s ** 2) / (num_shapes - 1)

    # Calculate variance ratios
    total_variance = np.sum(eigenvalues)
    variance_ratio = eigenvalues / total_variance

    # Find number of modes to retain
    cumulative_variance = np.cumsum(variance_ratio)
    num_modes = np.argmax(cumulative_variance >= variance_threshold) + 1

    # Extract reduced matrices
    U_reduced = centered_shapes.T @ U[:, :num_modes] / np.sqrt((num_shapes - 1) * eigenvalues[:num_modes])
    eigenvalues_reduced = eigenvalues[:num_modes]

    return mean_shape, U_reduced, eigenvalues_reduced, num_modes