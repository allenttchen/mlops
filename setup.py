#!/usr/bin/env python

from setuptools import setup, find_packages

ROOT_DIR = "mlops"
SOURCE_DIR = "src"

all_packages = ["src", "utils", "scripts"]
package_dir = {}
for package in all_packages:
    package_dir[package] = SOURCE_DIR
print(package_dir)
setup(
      name='mlops',
      version='1.0',
      description='MLOps pipeline',
      author='Allen Chen',
      author_email='allenttchen@gmail.com',
      #py_modules=['src', 'utils', 'scripts'],
      license='MIT',
      #url='',
      packages=all_packages,
      package_dir=package_dir,
)
