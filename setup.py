# -*- coding: utf-8 -*-
"""
Created on June 22 2026

@author: Karl Mayer
"""

from setuptools import setup, find_namespace_packages
import sys


def forbid_publish():
    """ Prevent users of setup.py from possibly publishing to PyPI.
    """
    # 'testarg' is for testing this command
    blacklist = ['register', 'upload', 'upload_docs', 'testarg',]

    for command in blacklist:
        if (command in sys.argv):
            err_str = "The input command \'" + command
            err_str += "has been blacklisted, exiting..."
            raise RuntimeError(err_str)
# This command must run before anything else in this file!
forbid_publish()


setup(
    name = 'circuit-benchmarks',
    packages = ['circuit_benchmarks'],
    include_package_data = True,
    description = 'Package for benchmarking quantum components and circuits, written in Guppy',
    version = '0.1.0',
    url = '',
    author = 'Karl Mayer',
    keywords = ['pip','circuit-benchmarks'],
    install_requires = [
          'matplotlib', 'numpy', 'scipy', 'guppylang', 'qnexus', 'pytket'],
    )
