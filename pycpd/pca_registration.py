import numbers

import numpy as np

from pycpd.emregistration import EMRegistration
from pycpd.utility import pca_kernel


class PCADeformableRegistration(EMRegistration):
    def __init__(self, X, Y, alpha=None, mean_shape=None, U=None, eigenvalues=None, *args, **kwargs):
        # Initialize EMRegistration first with its parameters
        super().__init__(X=X, Y=Y, *args, **kwargs)

        # Handle PCA-specific parameters
        if alpha is not None and (not isinstance(alpha, numbers.Number) or alpha <= 0):
            raise ValueError(
                "Expected a positive value for regularization parameter alpha. Instead got: {}".format(alpha))

        if mean_shape is None or U is None or eigenvalues is None:
            raise ValueError("mean_shape, U, and eigenvalues must be provided")

        self.alpha = 2 if alpha is None else alpha
        self.mean_shape = mean_shape
        self.U = U
        self.eigenvalues = eigenvalues
        self.W = np.zeros((self.M, self.D))
        self.prev_W = None
        self.G = pca_kernel(self.Y, self.mean_shape, self.U, self.eigenvalues)

        # Track both types of changes
        self.sigma_diff = np.inf
        self.w_diff = np.inf

    def update_transform(self):
        """
        Calculate a new estimate of the PCA-based deformable transformation.
        """
        A = np.dot(np.diag(self.P1), self.G) + self.alpha * self.sigma2 * np.eye(self.M)
        B = self.PX - np.dot(np.diag(self.P1), self.Y)

        # Store previous W
        self.prev_W = self.W.copy() if self.prev_W is None else self.W.copy()

        # Add stability term and solve
        A = A + 1e-8 * np.eye(len(A))
        self.W = np.linalg.solve(A, B)

        # Compute relative change in W
        self.w_diff = np.mean(np.abs(self.W - self.prev_W))
        # print(f"Iteration {self.iteration}: W change = {self.w_diff:.6f}")

    def transform_point_cloud(self, Y=None):
        """
        Update a point cloud using the new estimate of the PCA-based deformable transformation.
        """
        if Y is not None:
            G = pca_kernel(X=Y, mean_shape=self.mean_shape, U=self.U,
                           eigenvalues=self.eigenvalues, Y=self.Y)
            return Y + np.dot(G, self.W)
        else:
            prev_TY = self.TY.copy() if hasattr(self, 'TY') else self.Y.copy()
            self.TY = self.Y + np.dot(self.G, self.W)

            # Print debug info about transformation
            max_displacement = np.max(np.abs(self.TY - prev_TY))
            # print(f"Max displacement: {max_displacement:.6f}")

    def update_variance(self):
        """
        Update the variance of the mixture model using the new estimate of the transformation.
        """
        qprev = self.sigma2

        # Calculate distances in the transformed space
        xPx = np.dot(np.transpose(self.Pt1), np.sum(
            np.multiply(self.X, self.X), axis=1))
        yPy = np.dot(np.transpose(self.P1), np.sum(
            np.multiply(self.TY, self.TY), axis=1))
        trPXY = np.sum(np.multiply(self.TY, self.PX))

        # Update sigma2 with point matching error
        self.sigma2 = (xPx - 2 * trPXY + yPy) / (self.Np * self.D)

        # Add small regularization term
        self.sigma2 += 1e-8

        # Update sigma difference
        self.sigma_diff = np.abs(self.sigma2 - qprev)

        # Update overall difference using both criteria
        self.diff = max(self.sigma_diff / (self.sigma2 + 1e-8),
                        self.w_diff / (np.mean(np.abs(self.W)) + 1e-8))

        # print(
        #     f"Sigma2: {self.sigma2:.6f}, Sigma diff: {self.sigma_diff:.6f}, W diff: {self.w_diff:.6f}, Combined diff: {self.diff:.6f}")

    def get_registration_parameters(self):
        return self.G, self.W