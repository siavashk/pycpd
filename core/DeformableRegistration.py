import numpy as np
import matplotlib.pyplot as plt

class DeformableRegistration(object):
    def __init__(self, X, Y, alpha=None, beta=None, sigma2=None, maxIterations=100, tolerance=0.001, w=0):
        if X.shape[1] != Y.shape[1]:
            raise 'Both point clouds must have the same number of dimensions!'

        self.X             = X
        self.Y             = Y
        (self.N, self.D)   = self.X.shape
        (self.M, _)        = self.Y.shape
        self.alpha         = 2 if alpha is None else alpha
        self.beta          = 2 if alpha is None else beta
        self.W             = np.zeros((self.M, self.D))
        self.G             = np.zeros((self.M, self.M))
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

        return self.Y, np.dot(self.G, self.W)

    def iterate(self):
        self.EStep()
        self.MStep()
        self.iteration = self.iteration + 1

    def MStep(self):
        self.updateTransform()
        self.transformPointCloud()
        self.updateVariance()

    def updateTransform(self):
        A = np.dot(np.diag(self.P1), self.G) + self.alpha * self.sigma2 * np.eye(self.M)
        B = np.dot(self.P, self.X) - np.dot(np.diag(self.P1), self.Y)
        self.W = np.linalg.solve(A, B)

    def transformPointCloud(self, Y=None):
        if not Y:
            self.Y = self.Y + np.dot(self.G, self.W)
            return
        else:
            return Y + np.dot(self.G, self.W)

    def updateVariance(self):
        qprev = self.sigma2

        xPx      = np.dot(np.transpose(self.Pt1), np.sum(np.multiply(self.X, self.X), axis=1))
        yPy      = np.dot(np.transpose(self.P1),  np.sum(np.multiply(self.Y, self.Y), axis=1))
        trPXY    = np.sum(np.multiply(self.Y, np.dot(self.P, self.X)))

        self.sigma2 = (xPx - trPXY) / (self.Np * self.D)

        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10

        self.err = np.abs(self.sigma2 - qprev)

    def initialize(self):
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
        self._makeKernel()

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

    def _makeKernel(self):
        XX = np.reshape(self.Y, (1, self.M, self.D))
        YY = np.reshape(self.Y, (self.M, 1, self.D))
        XX = np.tile(XX, (self.M, 1, 1))
        YY = np.tile(YY, (1, self.M, 1))
        diff = XX-YY
        diff = np.sum(np.multiply(diff, diff), self.D)
        self.G = np.exp(-diff / (2 * self.beta))
