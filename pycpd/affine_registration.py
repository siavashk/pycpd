from builtins import super
import numpy as np
from .expectation_maximization_registration import expectation_maximization_registration

class affine_registration(expectation_maximization_registration):
    def __init__(self, B=None, t=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.B = np.eye(self.D) if B is None else B
        self.t = np.atleast_2d(np.zeros((1, self.D))) if t is None else t

    def update_transform(self):
        muX = np.divide(np.sum(np.dot(self.P, self.X), axis=0), self.Np)
        muY = np.divide(np.sum(np.dot(np.transpose(self.P), self.Y), axis=0), self.Np)

        self.XX = self.X - np.tile(muX, (self.N, 1))
        YY      = self.Y - np.tile(muY, (self.M, 1))

        self.A = np.dot(np.transpose(self.XX), np.transpose(self.P))
        self.A = np.dot(self.A, YY)

        self.YPY = np.dot(np.transpose(YY), np.diag(self.P1))
        self.YPY = np.dot(self.YPY, YY)

        self.B = np.linalg.solve(np.transpose(self.YPY), np.transpose(self.A))
        self.t = np.transpose(muX) - np.dot(np.transpose(self.B), np.transpose(muY))

    def transform_point_cloud(self, Y=None):
        if Y is None:
            self.TY = np.dot(self.Y, self.B) + np.tile(self.t, (self.M, 1))
            return
        else:
            return np.dot(Y, self.B) + np.tile(self.t, (Y.shape[0], 1))

    def update_variance(self):
        qprev = self.q

        trAB     = np.trace(np.dot(self.A, self.B))
        xPx      = np.dot(np.transpose(self.Pt1), np.sum(np.multiply(self.XX, self.XX), axis =1))
        trBYPYP  = np.trace(np.dot(np.dot(self.B, self.YPY), self.B))
        self.q   = (xPx - 2 * trAB + trBYPYP) / (2 * self.sigma2) + self.D * self.Np/2 * np.log(self.sigma2)
        self.err = np.abs(self.q - qprev)

        self.sigma2 = (xPx - trAB) / (self.Np * self.D)

        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10

    def get_registration_parameters(self):
        return self.B, self.t
