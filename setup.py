'''
Code for Fast k-Nearest Neighbour Search via Prioritized DCI

This code implements the method described in the Prioritized DCI paper, 
which can be found at https://arxiv.org/abs/1703.00440

This file is a part of the Dynamic Continuous Indexing reference 
implementation.


This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

Copyright (C) 2017    Ke Li
'''

import sys
from numpy.distutils.misc_util import Configuration
from numpy.distutils.system_info import get_info
from numpy.distutils.core import setup

def build_ext(config, dist):
    
    lapack_info = get_info('lapack_opt', 1)
    dci_sources = ['src/dci.c', 'src/py_dci.c', 'src/util.c']
    dci_headers = ['include/dci.h', 'include/util.h']
    if lapack_info:
        config.add_extension(name='_dci',sources=dci_sources, depends=dci_headers, include_dirs=['include'], extra_info=lapack_info, extra_compile_args=['-fopenmp'], extra_link_args=['-lgomp'])

    if not lapack_info:
        raise ImportError("No BLAS library found.")

    config_dict = config.todict()
    try:
        config_dict.pop('packages')
    except:
        pass

    return config_dict

def setup_dci(dist):

    config_dict = build_ext(Configuration('dciknn', parent_package=None, top_path=None), dist)
    
    setup(  version="0.1.0",
            description="Dynamic Continuous Indexing reference implementation.",
            author="Ke Li",
            author_email="ke.li@eecs.berkeley.edu",
            url="https://people.eecs.berkeley.edu/~ke.li/projects/dci",
            license="Mozilla Public License 2.0",
            classifiers=[
                'Development Status :: 4 - Beta',
                'Environment :: Console',
                'Operating System :: OS Independent',
                'Intended Audience :: Developers',
                'Intended Audience :: Science/Research',
                'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
                'Programming Language :: Python :: 2.7',
                'Programming Language :: Python :: 3.4',
                'Programming Language :: Python :: 3.5',
                'Programming Language :: Python :: 3.6',
                'Topic :: Scientific/Engineering :: Mathematics',
                'Topic :: Software Development :: Libraries :: Python Modules',
                 ],
            requires=['NumPy',],
            long_description="""
            Dynamic Continuous Indexing (DCI) is a family of randomized algorithms for
            exact k-nearest neighbour search that overcomes the curse of dimensionality.
            Its query time complexity is linear in ambient dimensionality and sublinear
            in intrinsic dimensionality. ``dciknn`` is a python package that contains
            the reference implementation of DCI and a convenient Python interface. 

            ``dciknn`` requires ``NumPy``. 
            """,
            packages=["dciknn"],
            **(config_dict))
            
if __name__ == '__main__':
    dist = sys.argv[1]
    setup_dci(dist)
