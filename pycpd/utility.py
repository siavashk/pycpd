import numpy as np


def is_positive_semi_definite(R):
    return np.all(np.linalg.eigvals(R) > 0)
