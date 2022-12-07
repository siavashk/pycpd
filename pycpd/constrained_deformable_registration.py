from builtins import super
import numpy as np
import numbers
from .deformable_registration import DeformableRegistration


class ConstrainedDeformableRegistration(DeformableRegistration):
    """
    Constrained deformable registration.

    Attributes
    ----------
    alpha: float (positive)
        Represents the trade-off between the goodness of maximum likelihood fit and regularization.

    beta: float(positive)
        Width of the Gaussian kernel.

    e_alpha: float (positive)
        Reliability of correspondence priors. Between 1e-8 (very reliable) and 1 (very unreliable)
    
    source_id: numpy.ndarray (int) 
        Indices for the points to be used as correspondences in the source array

    target_id: numpy.ndarray (int) 
        Indices for the points to be used as correspondences in the target array

    """

    def __init__(self, e_alpha = None, source_id = None, target_id= None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if e_alpha is not None and (not isinstance(e_alpha, numbers.Number) or e_alpha <= 0):
            raise ValueError(
                "Expected a positive value for regularization parameter e_alpha. Instead got: {}".format(e_alpha))
        
        if type(source_id) is not np.ndarray or source_id.ndim != 1:
            raise ValueError(
                "The source ids (source_id) must be a 1D numpy array of ints.")
        
        if type(target_id) is not np.ndarray or target_id.ndim != 1:
            raise ValueError(
                "The target ids (target_id) must be a 1D numpy array of ints.")

        self.e_alpha = 1e-8 if e_alpha is None else e_alpha
        self.source_id = source_id
        self.target_id = target_id
        self.P_tilde = np.zeros((self.M, self.N))
        self.P_tilde[self.source_id, self.target_id] = 1
        self.P1_tilde = np.sum(self.P_tilde, axis=1)
        self.PX_tilde = np.dot(self.P_tilde, self.X)

    def update_transform(self):
        """
        Calculate a new estimate of the deformable transformation.
        See Eq. 22 of https://arxiv.org/pdf/0905.2635.pdf.

        """
        if self.low_rank is False:
            A = np.dot(np.diag(self.P1), self.G) + \
                self.sigma2*(1/self.e_alpha)*np.dot(np.diag(self.P1_tilde), self.G) + \
                self.alpha * self.sigma2 * np.eye(self.M)
            B = self.PX - np.dot(np.diag(self.P1), self.Y) + self.sigma2*(1/self.e_alpha)*(self.PX_tilde - np.dot(np.diag(self.P1_tilde), self.Y)) 
            self.W = np.linalg.solve(A, B)

        elif self.low_rank is True:
            # Matlab code equivalent can be found here:
            # https://github.com/markeroon/matlab-computer-vision-routines/tree/master/third_party/CoherentPointDrift
            dP = np.diag(self.P1) + self.sigma2*(1/self.e_alpha)*np.diag(self.P1_tilde)
            dPQ = np.matmul(dP, self.Q)
            F = self.PX - np.dot(np.diag(self.P1), self.Y) + self.sigma2*(1/self.e_alpha)*(self.PX_tilde - np.dot(np.diag(self.P1_tilde), self.Y)) 

            self.W = 1 / (self.alpha * self.sigma2) * (F - np.matmul(dPQ, (
                np.linalg.solve((self.alpha * self.sigma2 * self.inv_S + np.matmul(self.Q.T, dPQ)),
                                (np.matmul(self.Q.T, F))))))
            QtW = np.matmul(self.Q.T, self.W)
            self.E = self.E + self.alpha / 2 * np.trace(np.matmul(QtW.T, np.matmul(self.S, QtW)))