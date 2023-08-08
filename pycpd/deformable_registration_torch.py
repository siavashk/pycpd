from __future__ import division
import numbers, torch, numpy as np
from warnings import warn
from pycpd.utility import *

def compute_sigma2(X, Y, P=None):
    """ Compute the variance (sigma2).

    Inputs:
        - X: torch.tensor (N,D). Target points
        - Y: torch.tensor (M,D). Source gmm centroids
        - P: torch.tensor (M,N). Soft centroid assignment matrix
    
    Returns:
        - sigma2 (M,). Per-point covariances
    """
    (N, D) = X.shape
    (M, _) = Y.shape
    if P is None: P = torch.ones(M,N).cuda()

    diff2 = torch.norm(Y[:,None,:] - X, dim=-1, p=2)**2  # (M,1,3) - (N,3) -> (M,N,3) -> (M,N)
    weighted_diff2 = P * diff2      # (M,N)
    denom = P.sum(dim=-1)[:,None]  # (M,1)
    sigma2 = torch.sum(weighted_diff2 / denom, dim=-1) / D  # (M, N) -> (M,)

    return sigma2


class DeformableRegistrationTorch(object):

    def __init__(self, X, Y, P=None, sigma2=None, max_iterations=None, tolerance=None, 
        alpha=2, beta=2, w=None, *args, **kwargs):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.X = X
        self.Y = Y
        self.TY = Y
        self.sigma2 = compute_sigma2(X, Y, P) if sigma2 is None else sigma2
        (self.N, self.D) = self.X.shape
        (self.M, _) = self.Y.shape
        self.tolerance = 0.001 if tolerance is None else tolerance
        self.w = 0.0 if w is None else w
        self.max_iterations = 100 if max_iterations is None else max_iterations
        self.iteration = 0
        self.diff = torch.inf
        self.q = torch.inf
        self.P = torch.zeros((self.M, self.N), device=self.device) if P is None else P
        self.Pt1 = torch.zeros((self.N, ), device=self.device)
        self.P1 = torch.zeros((self.M, ), device=self.device)
        self.PX = torch.zeros((self.M, self.D), device=self.device)
        self.Np = 0

        self.alpha = alpha; self.beta = beta
        self.W = torch.zeros((self.M, self.D), device=self.device)
        self.G = gaussian_kernel(self.Y, self.beta) # FIXME: Why is G only rank-15?
        
        
    def register(self, callback=lambda **kwargs: None):
        self.transform_point_cloud()
        while self.iteration < self.max_iterations and self.diff > self.tolerance:
            self.iterate()
            if callable(callback):
                kwargs = {'iteration': self.iteration,
                          'error': self.q, 'X': self.X, 'Y': self.TY}
                callback(**kwargs)

        return self.TY, self.get_registration_parameters()


    def iterate(self):
        """
        Perform one iteration of the EM algorithm.
        """
        self.expectation()
        self.maximization()
        self.iteration += 1

    def expectation(self):
        """
        Compute the expectation step of the EM algorithm.
        """
        P = torch.sum((self.X[None, :, :] - self.TY[:, None, :])**2, axis=2) # (M, N)
        c = 0
        
        if isinstance(self.sigma2, numbers.Number):
            P = torch.exp(-P/(2*self.sigma2))
            c = (2*torch.pi*self.sigma2)**(self.D/2)*self.w/(1. - self.w)*self.M/self.N
        else:
            P = torch.exp(-P/(2*self.sigma2[:,None]))
            c = (2*torch.pi*torch.mean(self.sigma2))**(self.D/2)*self.w/(1. - self.w)*self.M/self.N

        den = torch.sum(P, axis = 0, keepdims = True) # (1, N)
        den = torch.clip(den, torch.finfo(self.X.dtype).eps, None) + c

        self.P = torch.divide(P, den)
        self.Pt1 = torch.sum(self.P, axis=0)
        self.P1 = torch.sum(self.P, axis=1)
        self.Np = torch.sum(self.P1)
        self.PX = torch.matmul(self.P, self.X)

    def maximization(self):
        """
        Compute the maximization step of the EM algorithm.
        """
        self.update_transform()
        self.transform_point_cloud()
        self.update_variance()

    def update_transform(self):
        """
        Calculate a new estimate of the deformable transformation.
        See Eq. 22 of https://arxiv.org/pdf/0905.2635.pdf.

        """
        if isinstance(self.sigma2, numbers.Number) or self.sigma2.shape[0]==1:
            A = torch.diag(self.P1) @ self.G + \
                self.alpha * self.sigma2 * torch.eye(self.M) 
        
        else: 
            A = torch.diag(self.P1) @ self.G + \
                self.alpha * torch.diag(self.sigma2)
            
        B = self.PX - (torch.diag(self.P1) @ self.Y)

        self.W = torch.pinverse(A) @ B

 
    def transform_point_cloud(self, Y=None):
        if Y is not None:
            G = gaussian_kernel(X=Y, beta=self.beta, Y=self.Y)
            return Y + (G @ self.W)
        else:
            self.TY = self.Y + (self.G @ self.W)


    def update_variance(self):
        """
        Update the variance of the mixture model using the new estimate of the deformable transformation.
        See the update rule for sigma2 in Eq. 23 of of https://arxiv.org/pdf/0905.2635.pdf.

        """
        qprev = self.sigma2

        # Assume all \sigma_m^2 are the same
        if isinstance(self.sigma2, numbers.Number): 
            # The original CPD paper does not explicitly calculate the objective functional.
            # This functional will include terms from both the negative log-likelihood and
            # the Gaussian kernel used for regularization.
            self.q = torch.inf

            xPx = (self.Pt1.T) @ torch.sum(
                torch.multiply(self.X, self.X), axis=1)
            yPy = (self.P1.T) @ torch.sum(
                torch.multiply(self.TY, self.TY), axis=1)
            trPXY = torch.sum(torch.multiply(self.TY, self.PX))

            self.sigma2 = (xPx - 2 * trPXY + yPy) / (self.Np * self.D)

            if self.sigma2 <= 0:
                self.sigma2 = self.tolerance / 10
        
        # Assume each \sigma_m^2 is different
        else:   
            self.sigma2 = compute_sigma2(self.X, self.Y, self.P)

        # Here we use the difference between the current and previous
        # estimate of the variance as a proxy to test for convergence.    
        self.diff = torch.mean(torch.abs(self.sigma2 - qprev))

    def get_registration_parameters(self): return self.G, self.W


class DeformableRegistrationLoss(nn.Module):
    def __init__(self, alpha=2, beta=2):
        super(DeformableRegistrationLoss).__init__()
        self.alpha = alpha; self.beta = beta

    def forward(self, X, Y, sigma2, P, G, W):
        '''
        Inputs:
            - X: (N,D=3).       Target point cloud 
            - Y: (M,D).         Source gmm centroids
            - sigma2: (M,).     Variance of each gmm centroid
            - P: (M,N).         Soft cluster assignments
            - G: (M,M).         Y after gaussian kernel
            - W: (M,D).         Deformable transformation  \delta Y  = G @ W
        '''
        (N,D) = X.shape; (M,_) = Y.shape

        log_term = torch.log(sigma2)[:,None]                    # (M,1)
        diff_term = torch.norm(Y[:,None,:] - X, dim=-1, p=2)**2 # (M,N)
        diff_term /= 2 * log_term                               # (M,N)

        loss = (log_term * D/2) + diff_term                     # (M,N)
        pass
