#!/usr/bin/env python

from distutils.core import setup
setup(
    name = 'Sensor_Array',
    version = '0.1.0',
    description = 'Simulation and Analysis of MOF-based gas sensor arrays',
    author = 'Brian A Day, Jenna A Gustafson',
    author_email = 'brd84@pitt.edu',
    url = 'https://github.com/birdday/Sensor_Array',
    packages = ['sensor_array'],
    install_requires=[
        'numpy',
        'scipy',
        'click',
        'pandas',
        'pyyaml',
	'tensorflow',
    ],
    extras_require={'plotting': ['matplotlib']}
    tests_require=['pytest']
    entry_points={'console_scripts': []},
)
