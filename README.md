# Python-CPD

Pure Numpy Implementation of the Coherent Point Drift Algorithm.

MIT License.

## Introduction

This is a pure numpy implmenetation of the coherent point drift ([CPD](https://arxiv.org/abs/0905.2635)) algorithm by Myronenko and Song. It provides three registration methods for point clouds: 1) Scale and rigid registration; 2) Affine registration; and 3) Gaussian regularized non-rigid registration. 

I have only tested the methods for 2D point clouds, but they should work for 3D and higher dimensions as well.

## Installation

This repository only needs the packages listed in requirments.txt. I believe you can install them by calling:

`pip install -r requirements.txt`

## Usage

Each registration method is contained within a single class inside the core subfolder. To try out the registration, you can simply call `python tests/fish{Transform}.py`, where Transform is either Rigid, Affine or Deformable.

