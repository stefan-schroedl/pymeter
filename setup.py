from setuptools import setup

setup(
   name='meter',
   version='1.0',
   description='Simple utilities for stream statistics and timing',
   install_requires=[
       'numpy',
       'torch',
       'pytest'
       ],
   author='Stefan Schroedl',
)
