#############
Python-CPD
#############

.. |master_badge| image:: https://github.com/siavashk/pycpd/actions/workflows/build-test.yml/badge.svg?branch=master
.. |development_badge| image:: https://github.com/siavashk/pycpd/actions/workflows/build-test.yml/badge.svg?branch=development

+-----------------+---------------------+
| master          | |master_badge|      |
+-----------------+---------------------+
| development     | |development_badge| |
+-----------------+---------------------+


`Documentation <https://siavashk.github.io/pycpd/>`_

Pure Numpy Implementation of the Coherent Point Drift Algorithm.

MIT License.

*************
Introduction
*************

This is a pure numpy implementation of the coherent point drift `CPD <https://arxiv.org/abs/0905.2635/>`_ algorithm by Myronenko and Song for use by the python community. It provides three registration methods for point clouds: 1) Scale and rigid registration; 2) Affine registration; and 3) Gaussian regularized non-rigid registration.

The CPD algorithm is a registration method for aligning two point clouds. In this method, the moving point cloud is modelled as a Gaussian Mixture Model (GMM) and the fixed point cloud are treated as observations from the GMM. The optimal transformation parameters maximze the Maximum A Posteriori (MAP) estimation that the observed point cloud is drawn from the GMM.

The registration methods work for arbitrary MxN 2D arrays where M is the number of "points" and N is the number of dimensions. A typical point cloud would be Mx2 or Mx3 for 2D and 3D points clouds respectively. For more information, please refer to my `blog <http://siavashk.github.io/2017/05/14/coherent-point-drift/>`_.

*************
Installation
*************

Install from PyPI
#################

.. code-block:: bash

  pip install pycpd

Installation from Source
########################

Clone the repository to a location, referred to as the ``root`` folder. For example:

.. code-block:: bash

  git clone https://github.com/siavashk/pycpd.git $HOME/pycpd

Install the package:

.. code-block:: bash

  pip install .

or 

.. code-block:: bash

  make requirements
  make build

Install Matplotlib for Visualization
####################################

For running sample registration examples under ``/examples``, you will need ``Matplotlib`` to visualize the registration. This can be downloaded by running:

.. code-block:: bash

 pip install matplotlib

or 

.. code-block:: bash

  make visualize
  
*****
Usage
*****

Each registration method is contained within a single class inside the ``pycpd`` subfolder. To try out the registration, you can simply run:

.. code-block:: bash

python examples/fish_{Transform}_{Dimension}.py

where ``Transform`` is either ``rigid``, ``affine`` or ``deformable`` and ``Dimension`` is either ``2D`` or ``3D``. Note that examples are meant to be run from the ``root`` folder.

********
Example
********

Basic Usage
###########

Basic usage includes providing any of the registration methods with 2 arrays that are MxN & BxN. E.g., they can have different numbers of points (M & B) but must have the same number of dimensions per point (N).

.. code-block:: python

  from pycpd import RigidRegistration
  import numpy as np

  # create 2D target points (you can get these from any source you desire)
  # creating a square w/ 2 additional points. 
  target = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0], [0, 0.5]])
  print('Target Points: \n', target)

  # create a translation to apply to the target for testing the registration
  translation = [1, 0]

  # create a fake source by adding a translation to the target.
  # in a real use, you would load the source points from a file or other source. 
  # the only requirement is that this array also be 2-dimensional and that the 
  # second dimension be the same length as the second dimension of the target array.
  source = target + translation
  print('Source Points: \n', source)

  # create a RigidRegistration object
  reg = RigidRegistration(X=target, Y=source)
  # run the registration & collect the results
  TY, (s_reg, R_reg, t_reg) = reg.register()

  # TY is the transformed source points
  # the values in () are the registration parameters.
  # In this case of rigid registration they are:
  #     s_reg the scale of the registration
  #     R_reg the rotation matrix of the registration
  #     t_reg the translation of the registration


The affine and deformable registration methods are used in the same way, but provide their respective transformation parameters.

Apply Transform to Another Point Cloud
#######################################
Sometimes you may want to apply the transformation parameters to another point cloud. For example, if you have a very large point cloud
it is sometimes appropriate to randomly sample some of the points for registration and then apply the transformation to the entire point cloud. 

To do this, after fitting the above registration, you would run `reg.transform_point_cloud(Y=points_to_transform)`. This will apply the learned 
registration parameters to the point cloud `points_to_transform` and return the transformed point cloud.

Tuning Registration parameters
##############################

For rigid and affine registrations the main parameter you can tweak is `w`. The `w` parameter is an indication of the amount of noise in the 
point clouds `[0,1]`, by default it is set to `0` assuming no noise, but can be set at any value `0 <= w <1` with higher values indicating more noise. 

For deformable registration, you can also tune `alpha`, `beta`, and use `low_rank`. 

The `alpha` parameter (`lambda` in the original paper) identifies a tradeoff between making points align & regularization of the deformation. 
A higher value makes the deformation more rigid, a lower value makes the deformation more flexible. 

The `beta` is the width of the Gaussian kernel used to regularize the deformation and thus identifies how far apart points should be
to move them together (coherently). `beta` depends on the scale/size of your points cloud. Tuning `beta` can be simplified by normalizing 
the point cloud to a unit sphere distance.

The `low_rank` parameter is a boolean that indicates whether to use a regularized form of the deformation field. This further
constrains the deformation, while vastly speeding up the optimization. `num_eig` is the number of eigenvalues to use in the low rank 
approximation. `num_eig` should be less than the number of points in the point cloud, the lower the smoother the deformation and the
faster the optimization.



*******
Testing
*******

Tests can be run using pytest:

.. code-block:: bash

 pip install pytest
 pytest

or 

.. code-block:: bash
  
  make dev
  make test

*************
Documentation
*************

The documentation can be built using pydoc3

.. code-block:: bash
  
  make dev
  make doc

************
Contributing
************

Contributions are welcome. Please see the guidelines outlined in the document: `CONTRIBUTING <https://github.com/siavashk/pycpd/blob/master/CONTRIBUTING.md>`_.

***************
Code of Conduct
***************
We have adopted the code of conduct defined by the `Contributor Covenant <https://www.contributor-covenant.org/>`_ to clarify expected behavior in our community. For more information see the `Code of Conduct <https://github.com/siavashk/pycpd/blob/master/CODE_OF_CONDUCT.md>`_.