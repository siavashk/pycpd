---
title: 'PyCPD: Pure NumPy Implementation of the Coherent Point Drift Algorithm'
tags:
  - Python
  - Point-cloud
  - Registration
  - Expectation-Maximization

authors:
  - name: Anthony A. Gatti 
    orcid: 0000-0001-6717-8979
    affiliation: "1, 2" 
  - name: Siavash Khallaghi^[Corresponding author] 
    affiliation: 3                                    
affiliations:
 - name: Stanford University, USA
   index: 1
 - name: NeuralSeg Ltd., Canada
   index: 2
 - name: Independent Researcher, Canada          # Siavash - DO YOU WANT TO INCLUDE YOUR CURRENT EMPLOYER? 
   index: 3
date: March 31 2022
bibliography: paper.bib

---

# Background
Point cloud registration is a common problem in many areas of computer science,
particularly computer vision. Point clouds come from many types of data such 
as LIDAR commonly used for self-driving vehicles, and other sorts of 3D scanners 
(e.g., structured light) are commonly used to map the surface of physical objects.
Point clouds are also used to represent the surface of an anatomical structure
extracted from a medical image. Point cloud registration finds a transformaton 
from one point cloud to another. Point cloud registration has use cases in many 
fields from self-driving vehicles to medical imaging and virtual reality. 
Typically, point cloud registraton is classified into rigid (only rotations or
translations), affine (rigid + shearing and scaling) and non-rigid also called 
deformable registration (non-linear deformation). 

Point cloud registration typically requires 2 point clouds. The first point 
cloud is the "fixed" or "target" point cloud and the second is the "moving" 
or "source" point cloud. We try to find the transformation that will best
align the moving (or source) point cloud with the fixed point cloud. One 
of the most well known rigid point cloud registration algorithms is 
the Iterative Closest Point (ICP) algorithm [@121791; @CHEN1992145]. ICP is 
an iterative algorithm where the following steps are iterated: 

  (1) for every point on the moving point cloud find the closest point on the 
  fixed point cloud
  (2) use a least squares approach to find the optimal transformation matrix 
  (rotation, tranlsation, scaling, shear) to align the point correspondences
  found in (1)
  (3) apply the transformation from (2) to the moving point cloud

These steps are repeated until the root mean squared point-to-point distances
from (1) converges. 

The coherent point drift (CPD) algorithm was created by Myronenko and Song 
[@5432191] to overcome many of the limitaitons of ICP and other previous registration
methods. Namely, these other methods didnt necessarily generalize to greater than 
3 dimensions and they were prone to errors such as noise, outliers, or missing 
points. The CPD alogirthm is a probabilistic multidimensional algorithm that is 
robust and works for both rigid and non-rigid registration. In CPD the moving 
point cloud is modelled as a Gaussian Mixture Model (GMM) and the fixed point 
cloud is treated as observations from the GMM. The optimal transformation 
parameters maximze the Maximum Likelihood / Maximum A Posteriori (MAP) 
estimation that the observed point cloud is drawn from the GMM. A key point of
the CPD algorithm is that it forces the points to move coherently by preserving 
topological structure. The CPD algorithm is also an iterative algorithm that 
iterates between an expectation (E) step and a maximization (M) step until 
convergence is achieved. The E-step estimates the posterior probability 
distributions of the GMM centroids (moving points) given the data (fixed 
points) then the M-step updates the transfomration to maximize the posterior
probability that the data belong to the GMM distributions. The E- and M-steps 
are iterated until convergence.

# Statement of need
Due to the robustness and the broad array of uses for the CPD algorithm 
the original CPD paper has currently (March 2022) been referenced >2000 
times. The CPD algorithm is available in Matlab. However, no open-source
python version previously existed. In this paper we present a pure 
NumPy[@harris2020array] version of the CPD algorithm to enable general 
use of CPD for the Python community. Furthermore, the full implementation 
in Numpy makes the algorithm accesible for others to learn from. To help 
in learning, a blog post that coincides with this library has previously 
been [published](http://siavashk.github.io/2017/05/14/coherent-point-drift/)
[@khallaghi_2017].

# Summary
The PyCPD package implements the CPD algorithm in NumPy. The library itself 
includes a module to implement the Expectation Maximization (EM) algorithm. 
Sub-modules inherent the EM functionality and implement rigid, affine, and 
deformable registration using EM. CPD registration using affine, rigid, 
and deformable methods all allow for the transformation learned from CPD 
to be applied to any point cloud. Thus, it is possible to learn the 
transformation on a subset of the points and then apply it to the whole 
point cloud to reduce computation time. Finally, the low-rank approximation
for deformable registration that was described by Myronenko and Song 
[@5432191] was implemented. A low rank approximation of the Gaussian kernal 
is used to reduce computation time and has the added benefit of regularizing 
the non-rigid deformation. 

![Visualization of the 3D rigid registration from the examples included in the library. Each panel represents a different iteration in the registration process.](rigid_bunny_3d_registration.tiff)

Examples of the PyCPD algorithm are included (**Figure 1**). Examples are available for
2D and 3D versions of all registration methods (rigid, affine, deformable). 
Examples of how to use the low-rank approximation as well as how to use 
a sub-set of the points for registration are also included in the examples. 


# Acknowledgements

We acknowledge contributions from: 
  - Alvin Wan for testing on Python 3.x.
  - Talley Lambert for pointing out that the moving point cloud should be immutable during registration, 
  - Matthew DiFranco for fixing the check for target point clouds.
  - normanius for pointing out that the contribution of uniform distribution was not being added in the E-step.
  - Kai Zhang for finding a bug when transforming a point cloud using rigid registration parameters.
  - sandyhsia for finding a bug when updating the variance during deformable registration.

# References
