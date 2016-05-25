#!/usr/bin/python3
from setuptools import setup, find_packages

short_description = 'Common Graph Algorithms Library'

long_description = short_description +  '''
-------------------------------------

This is a library of common graph algorithms with a functional API so that it 
can directly work with arbitrary python data structures.

Currently Implemented:
- depth_first_traversal()       A lazy depth first traversal
- depth_first_search()          A depth first search
'''

setup(
    name = 'common_algorithms',
    version = "0.0.0",

    author = "David McDougall",
    author_email = "dam1784[at]rit[dot]edu",
    url = 'no-website',

    license = 'MIT',

    description = short_description,
    long_description = long_description,

    keywords = 'developement',

    classifiers = [
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
    ],

    packages = find_packages(),
)
