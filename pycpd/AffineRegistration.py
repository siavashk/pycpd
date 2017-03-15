import numpy as np
import matplotlib.pyplot as plt

class AffineRegistration(object):
    def __init__(self, X, Y, B=None, t=None, sigma2=None, maxIterations=100, tolerance=0.001, w=0):
        if X.shape[1] != Y.shape[1]:
            raise 'Both point clouds must have the same number of dimensions!'

        self.X             = X
        self.Y             = Y
        (self.N, self.D)   = self.X.shape
        (self.M, _)        = self.Y.shape
        self.B             = np.eye(self.D) if B is None else B
        self.t             = np.atleast_2d(np.zeros((1, self.D))) if t is None else t
        self.sigma2        = sigma2
        self.iteration     = 0
        self.maxIterations = maxIterations
        self.tolerance     = tolerance
        self.w             = w
        self.q             = 0
        self.err           = 0

    def register(self, callback):
        self.initialize()

        while self.iteration < self.maxIterations and self.err > self.tolerance:
            self.iterate()
            callback(X=self.X, Y=self.Y)

        return self.Y, self.B, self.t

    def iterate(self):
        self.EStep()
        self.MStep()
        self.iteration = self.iteration + 1

    def MStep(self):
        self.updateTransform()
        self.transformPointCloud()
        self.updateVariance()

    def updateTransform(self):
        muX = np.divide(np.sum(np.dot(self.P, self.X), axis=0), self.Np)
        muY = np.divide(np.sum(np.dot(np.transpose(self.P), self.Y), axis=0), self.Np)

        self.XX = self.X - np.tile(muX, (self.N, 1))
        YY      = self.Y - np.tile(muY, (self.M, 1))

        self.A = np.dot(np.transpose(self.XX), np.transpose(self.P))
        self.A = np.dot(self.A, YY)

        self.YPY = np.dot(np.transpose(YY), np.diag(self.P1))
        self.YPY = np.dot(self.YPY, YY)

        Bt = np.linalg.solve(np.transpose(self.YPY), np.transpose(self.A))
        self.B = np.transpose(Bt)
        self.t = np.transpose(muX) - np.dot(self.B, np.transpose(muY))

    def transformPointCloud(self, Y=None):
        if not Y:
            self.Y = np.dot(self.Y, np.transpose(self.B)) + np.tile(np.transpose(self.t), (self.M, 1))
            return
        else:
            return np.dot(Y, np.transpose(self.B)) + np.tile(np.transpose(self.t), (self.M, 1))

    def updateVariance(self):
        qprev = self.q

        trAB     = np.trace(np.dot(self.A, np.transpose(self.B)))
        xPx      = np.dot(np.transpose(self.Pt1), np.sum(np.multiply(self.XX, self.XX), axis =1))
        trBYPYP  = np.trace(np.dot(np.dot(self.B, self.YPY), np.transpose(self.B)))
        self.q   = (xPx - 2 * trAB + trBYPYP) / (2 * self.sigma2) + self.D * self.Np/2 * np.log(self.sigma2)
        self.err = np.abs(self.q - qprev)

        self.sigma2 = (xPx - trAB) / (self.Np * self.D)

        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10

    def initialize(self):
        self.Y = np.dot(self.Y, np.transpose(self.B)) + np.repeat(self.t, self.M, axis=0)
        if not self.sigma2:
            XX = np.reshape(self.X, (1, self.N, self.D))
            YY = np.reshape(self.Y, (self.M, 1, self.D))
            XX = np.tile(XX, (self.M, 1, 1))
            YY = np.tile(YY, (1, self.N, 1))
            diff = XX - YY
            err  = np.multiply(diff, diff)
            self.sigma2 = np.sum(err) / (self.D * self.M * self.N)

        self.err  = self.tolerance + 1
        self.q    = -self.err - self.N * self.D/2 * np.log(self.sigma2)

    def EStep(self):
        P = np.zeros((self.M, self.N))

        for i in range(0, self.M):
            diff     = self.X - np.tile(self.Y[i, :], (self.N, 1))
            diff    = np.multiply(diff, diff)
            P[i, :] = P[i, :] + np.sum(diff, axis=1)

        c = (2 * np.pi * self.sigma2) ** (self.D / 2)
        c = c * self.w / (1 - self.w)
        c = c * self.M / self.N

        P = np.exp(-P / (2 * self.sigma2))
        den = np.sum(P, axis=0)
        den = np.tile(den, (self.M, 1))
        den[den==0] = np.finfo(float).eps

        self.P   = np.divide(P, den)
        self.Pt1 = np.sum(self.P, axis=0)
        self.P1  = np.sum(self.P, axis=1)
        self.Np  = np.sum(self.P1)
