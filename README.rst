#############
Python-CPD
#############
.. image:: https://travis-ci.com/siavashk/pycpd.svg?branch=master
    :target: https://travis-ci.com/siavashk/pycpd

Pure Numpy Implementation of the Coherent Point Drift Algorithm.

MIT License.

*************
Introduction
*************

This is a pure numpy implementation of the coherent point drift `CPD <https://arxiv.org/abs/0905.2635/>`_ algorithm by Myronenko and Song. It provides three registration methods for point clouds: 1) Scale and rigid registration; 2) Affine registration; and 3) Gaussian regularized non-rigid registration.

The CPD algorithm is a registration method for aligning two point clouds. In this method, the moving point cloud is modelled as a Gaussian Mixture Model (GMM) and the fixed point cloud are treated as observations from the GMM. The optimal transformation parameters maximze the Maximum A Posteriori (MAP) estimation that the observed point cloud is drawn from the GMM.

The registration methods work for 2D and 3D point clouds. For more information, please refer to my `blog <http://siavashk.github.io/2017/05/14/coherent-point-drift/>`_.

*************
Pip Install
*************
.. code-block:: bash

  pip install pycpd

************************
Installation From Source
************************

Clone the repository to a location, referred to as the ``root`` folder. For example:

.. code-block:: bash

  git clone https://github.com/siavashk/pycpd.git $HOME/pycpd

Install the package:

.. code-block:: bash

  pip install .

For running sample registration examples under ``examples``, you will need ``matplotlib`` to visualize the registration. This can be downloaded by running:

.. code-block:: bash

 pip install matplotlib

*****
Usage
*****

Each registration method is contained within a single class inside the ``pycpd`` subfolder. To try out the registration, you can simply run:

.. code-block:: bash

 python examples/fish_{Transform}_{Dimension}.py

where ``Transform`` is either ``rigid``, ``affine`` or ``deformable`` and ``Dimension`` is either ``2D`` or ``3D``. Note that examples are meant to be run from the ``root`` folder.
