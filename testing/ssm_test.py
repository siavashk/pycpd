import numpy as np
import matplotlib.pyplot as plt

import sys
import os

# Add the parent directory of "scripts" to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pycpd import ssm

# define 2D shapes
# Example shape data (M shapes, 10 points, 2D)
M = 10  # Number of shapes
num_points = 3  # Number of points per shape
dimension = 2
shapes = np.random.rand(M, num_points, dimension)  # 10 x 3 x 2
threshold = 0.95

print("shapes: \n", shapes)

mean_shape, U_reduced, eigenvalues, num_modes = ssm.build_ssm(shapes, threshold)

print("mean_shape: \n", mean_shape)
print("U_reduced: \n", U_reduced)
print("eigenvalues: \n", eigenvalues)
print("num_modes: \n", num_modes)
