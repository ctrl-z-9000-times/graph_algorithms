#!/usr/bin/python3
from setuptools import setup

import graph_algorithms

with open('graph_algorithms/LICENSE') as f:
    license = f.read()

with open('README.txt') as f:
    readme = f.read()

setup(
    name = graph_algorithms.__name__,
    version = graph_algorithms.__version__,

    author = graph_algorithms.__author__,
    author_email = "dam1784[at]rit.edu",
    url = 'https://pypi.python.org/pypi/common_algorithms',

    license = license,

    description = graph_algorithms.__doc__.split('\n')[0],
    long_description = readme,

    keywords = ['development', 'graph'],

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
        "Programming Language :: Python :: 3.5",
    ],

    packages = ['graph_algorithms'],
    package_data = {
        '': '*LICENSE',
    },

    test_suite = "graph_algorithms.tests",

    extras_require = {
        'debug': ['numpy'],
        'test': ['numpy'],
    },
)
