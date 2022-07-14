"""
This is a pure numpy implementation of the coherent point drift [CPD](https://arxiv.org/abs/0905.2635/)
algorithm by Myronenko and Song. It provides three registration methods for point clouds: 

1. Scale and rigid registration
2. Affine registration
3. Gaussian regularized non-rigid registration

Licensed under an MIT License (c) 2010-2016 Siavash Khallaghi.
Distributed here: https://github.com/siavashk/pycpd
"""

from .rigid_registration import RigidRegistration
from .affine_registration import AffineRegistration
from .deformable_registration import gaussian_kernel, DeformableRegistration
