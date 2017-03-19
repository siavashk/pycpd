#############
Python-CPD
#############

Pure Numpy Implementation of the Coherent Point Drift Algorithm.

MIT License.

*************
Introduction
*************

This is a pure numpy implmenetation of the coherent point drift `CPD <https://arxiv.org/abs/0905.2635/>`_ algorithm by Myronenko and Song. It provides three registration methods for point clouds: 1) Scale and rigid registration; 2) Affine registration; and 3) Gaussian regularized non-rigid registration.

The CPD algorithm is a registration method for aligning two point clouds. In this method, the moving point cloud is modelled as a Gaussian Mixture Model (GMM) and the fixed point cloud are treated as observations from the GMM. The optimal transformation parameters maximze the Maximum A Posteriori (MAP) estimation that the observed point cloud is drawn from the GMM.

The registration methods work for 2D and 3D point clouds.

*************
Pip Install
*************
.. code-block:: bash

  $ pip install pycpd

************************
Installation From Source
************************

Clone the repository to a location in your home directory. For example:

.. code-block:: bash

  $ git clone https://github.com/siavashk/pycpd.git $HOME/pycpd

Install the package:

.. code-block:: bash

  $ pip install .

For running sample registration examples under `/tests`, you will need two additional packages.

Scipy (for loading `.mat` files) and matplotlib (for visualizing the reigstration). These can be downloaded by running:

.. code-block:: bash

 $ pip install -r requirements.txt

*****
Usage
*****

Each registration method is contained within a single class inside the pycpd subfolder. To try out the registration, you can simply call:

.. code-block:: bash

 $ python tests/fish{Transform}{Dimension}.py

where Transform is either Rigid, Affine or Deformable and Dimension is either 2D or 3D.
