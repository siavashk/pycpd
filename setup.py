from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='pycpd',
      version='2.0.0',
      description='Pure Numpy Implementation of the Coherent Point Drift Algorithm',
      long_description=readme(),
      url='https://github.com/siavashk/pycpd',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering',
      ],
      keywords='image processing, point cloud, registration, mesh, surface',
      author='Siavash Khallaghi',
      author_email='siavashk@ece.ubc.ca',
      license='MIT',
      packages=['pycpd'],
      install_requires=['numpy', 'future'],
      zip_safe=False)
