#!/usr/bin/env python

from codecs import open
from os import path
from distutils.core import setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='DataInsight',
      version='1.0',
      description='Machine Learning Toolkit',
      long_description=long_description,
      author='Christian Contreras-Campana',
      author_email='chrisjcc.physics@gmail.com',
      url='https://github.com/chrisjcc/DataInsight',
      packages=['neural_network', 'visualization']
)
