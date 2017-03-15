# Python-CPD

Pure Numpy Implementation of the Coherent Point Drift Algorithm.

MIT License.

## Introduction

This is a pure numpy implmenetation of the coherent point drift ([CPD](https://arxiv.org/abs/0905.2635)) algorithm by Myronenko and Song. It provides three registration methods for point clouds: 1) Scale and rigid registration; 2) Affine registration; and 3) Gaussian regularized non-rigid registration.

The registration methods work for 2D and 3D point clouds.

## Pip Install

Coming soon.

## Installation From Source

Clone the repository to a location in your home directory. For example:

`git clone https://github.com/siavashk/pycpd.git $HOME/pycpd`.

This repository only needs the packages listed in requirments.txt. I believe you can install them by calling:

`pip install -r requirements.txt`

Append `pycpd/core` to your `PYTHONPATH`:

`export PYTHONPATH=$PYTHONPATH:$HOME/pycpd/core`

## Usage

Each registration method is contained within a single class inside the core subfolder. To try out the registration, you can simply call `python tests/fish{Transform}{Dimension}.py`, where Transform is either Rigid, Affine or Deformable and Dimension is either 2D or 3D.
