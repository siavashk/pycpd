import numpy as np
from functools import reduce

class jrmpc_rigid(object):
  def __init__(self, Y, R=None, t=None, maxIterations=100, gamma=0.1, ):
    if Y is None:
      raise 'Empty list of point clouds!'

    dimensions = [cloud.shape[1] for cloud in Y]

    if not all(dimension == dimensions[0] for dimension in dimensions):
      raise 'All point clouds must have the same number of dimensions!'

    self.Y = Y
    self.M = [cloud.shape[0] for cloud in self.Y]
    self.D = dimensions[0]

    if R:
      rotations = [rotation.shape for rotation in R]
      if not all(rotation[0] == self.D and rotation[1] == self.D for rotation in rotations):
        raise 'All rotation matrices need to be %d x %d matrices!' % (self.D, self.D)
      self.R = R
    else:
      self.R = [np.eye(self.D) for cloud in Y]

    if t:
      translations = [translations.shape for translation in t]
      if not all(translations[0] == 1 and translations[1] == self.D for translation in translations):
        raise 'All translation vectors need to be 1 x %d matrices!' % (self.D)
      self.t = t
    else:
      self.t = [np.atleast_2d(np.zeros((1, self.D))) for cloud in self.Y]

  def initializeGMMMeans(self):
    self.K = np.median(self.M)
    az = 2 * np.pi * np.random.rand(self.K)
    el = 2 * np.pi * np.random.rand(self.K)
    self.X = np.array([np.multiply(np.cos(az), np.cos(el)), np.sin(el), np.multiply(np.sin(az), np.cos(el))])

  def print_self(self):
    print('Y has %d point clouds.' % (len(self.Y)))
    print('Each point cloud has M points: ', self.M)
    print('Dimensionality of all point clouds is: ', self.D)
    print('Rotation matrices are: ', self.R)
    print('Translation vectors are: ', self.t)
