#!/usr/bin/env python
"""
setup.py: GANdy project module repository
"""
import os
import sys
import time
from setuptools import setup, find_packages

__author__ = "Samantha Tetef, Kyle Moskowitz, Yu-Chi Fang, Yuxuan Ren, Evan Komp"
__copyright__ = "LICENSE.txt"
__version__ = "0.1.0"

setup(
    name='gandy',
    version=__version__,
    url='https://github.com/GANdy-team/GANdy',
    author=__author__,
    classifiers=[
        'Development Status :: 1 - Planning',
        'Environment :: Console',
        'Operating System :: OS Independant',
        'Programming Language :: Python',
	'Topic :: Scientific/Engineering'
    ],
    license=__copyright__,
    description='High level tool for creating supervised machine learning models capable of estimating prediction uncertainty.',
    keywords=[
        'GAN',
        'uncertainty',
        'GP',
        'BNN',
	'machine_learning',
    ],
    packages=['gandy'],
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.19.1'
    ]
)
