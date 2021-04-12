#!/usr/bin/env python3

from distutils.core import setup

setup(name='pipeline',
      version='1.0',
      description='Data pipeline for "Limitations of snapshot hyperspectral cameras to monitor plant response dynamics in stress-free conditions"',
      author='Olivier Pieters',
      author_email='olivier.pieters@ugent.be',
      packages=['pipeline'],
      package_data={"pipeline": ["data/*.csv"]},
      install_requires=["scikit-learn==0.23.2", "PyYAML==5.4", "scipy==1.5.2", "pandas==1.1.0"]
     )
