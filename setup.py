from setuptools import setup

setup(name='pycpd',
      version='0.1',
      description='Pure Numpy Implementation of the Coherent Point Drift Algorithm',
      long_description='The Coherent Point Drift (CPD) algorithm is a registration method for aligning two point clouds. In this method, the moving point cloud is modelled as a Gaussian Mixture Model (GMM) and the fixed point cloud are treated as observations from the GMM. The optimal transformation parameters maximze the Maximum A Posteriori (MAP) estimation that the observed point cloud is drawn from the GMM.',
      url='https://github.com/siavashk/pycpd',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering',
      ],
      keywords='image processing, point cloud, registration, mesh, surface',
      author='Siavash Khallaghi',
      author_email='siavashk@ece.ubc.ca',
      license='MIT',
      packages=['pycpd'],
      install_requires=['numpy'],
      zip_safe=False)
