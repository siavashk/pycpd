import numpy as np

def initialize_sigma2(X, Y):
    (N, D) = X.shape
    (M, _) = Y.shape
    XX = np.reshape(X, (1, N, D))
    YY = np.reshape(Y, (M, 1, D))
    XX = np.tile(XX, (M, 1, 1))
    YY = np.tile(YY, (1, N, 1))
    diff = XX - YY
    err  = np.multiply(diff, diff)
    return np.sum(err) / (D * M * N)

class expectation_maximization_registration(object):
    def __init__(self, X, Y, sigma2=None, max_iterations=100, tolerance=0.001, w=0, *args, **kwargs):
        if type(X) is not np.ndarray or X.ndim != 2:
            raise ValueError("The target point cloud (X) must be at a 2D numpy array.")
        if type(Y) is not np.ndarray or Y.ndim != 2:
            raise ValueError("The source point cloud (Y) must be a 2D numpy array.")
        if X.shape[1] != Y.shape[1]:
            raise ValueError("Both point clouds need to have the same number of dimensions.")

        self.X              = X
        self.Y              = Y
        self.sigma2         = sigma2
        (self.N, self.D)    = self.X.shape
        (self.M, _)         = self.Y.shape
        self.tolerance      = tolerance
        self.w              = w
        self.max_iterations = max_iterations
        self.iteration      = 0
        self.err            = self.tolerance + 1
        self.P              = np.zeros((self.M, self.N))
        self.Pt1            = np.zeros((self.N, ))
        self.P1             = np.zeros((self.M, ))
        self.Np             = 0

    def register(self, callback=lambda **kwargs: None):
        self.transform_point_cloud()
        if self.sigma2 is None:
            self.sigma2 = initialize_sigma2(self.X, self.TY)
        self.q = -self.err - self.N * self.D/2 * np.log(self.sigma2)
        while self.iteration < self.max_iterations and self.err > self.tolerance:
            self.iterate()
            if callable(callback):
                kwargs = {'iteration': self.iteration, 'error': self.err, 'X': self.X, 'Y': self.TY}
                callback(**kwargs)

        return self.TY, self.get_registration_parameters()

    def get_registration_parameters(self):
        raise NotImplementedError("Registration parameters should be defined in child classes.")

    def iterate(self):
        self.expectation()
        self.maximization()
        self.iteration += 1

    def expectation(self):
        P = np.zeros((self.M, self.N))

        for i in range(0, self.M):
            diff     = self.X - np.tile(self.TY[i, :], (self.N, 1))
            diff     = np.multiply(diff, diff)
            P[i, :]  = P[i, :] + np.sum(diff, axis=1)

        c = (2 * np.pi * self.sigma2) ** (self.D / 2)
        c = c * self.w / (1 - self.w)
        c = c * self.M / self.N

        P = np.exp(-P / (2 * self.sigma2))
        den = np.sum(P, axis=0)
        den = np.tile(den, (self.M, 1))
        den[den==0] = np.finfo(float).eps
        den += c

        self.P   = np.divide(P, den)
        self.Pt1 = np.sum(self.P, axis=0)
        self.P1  = np.sum(self.P, axis=1)
        self.Np  = np.sum(self.P1)

    def maximization(self):
        self.update_transform()
        self.transform_point_cloud()
        self.update_variance()
