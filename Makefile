# Code for Fast k-Nearest Neighbour Search via Prioritized DCI
#
# This code implements the method described in the Prioritized DCI paper, 
# which can be found at https://arxiv.org/abs/1703.00440
#
# This file is a part of the Dynamic Continuous Indexing reference 
# implementation.
#
# 
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (C) 2017    Ke Li
#
# Prerequisites:
# 1. A C compiler with support for OpenMP, e.g.: gcc
# 2. Python 2.7+ or Python 3.1+
# 3. A BLAS library (supported implementations include the reference 
#      implementation from Netlib, ATLAS, OpenBLAS and MKL)
# 4. Python development headers (can be installed using 
#      "apt-get install python2.7-dev" or "apt-get install python3.5-dev")
# 5. (If Python interface is desired) NumPy
# 6. (If TensorFlow op is desired) TensorFlow and C++ compiler

# Build Instructions:
# (All instructions below assume you are using Linux; please adjust accordingly
# if you are using Windows.)
#
# 1.  Set the BLAS variable to "netlib" (for the reference implementation, 
#       which will be referred to as Netlib), "atlas" (for ATLAS), 
#       "openblas" (for OpenBLAS) or "mkl" (for MKL). OpenBLAS and MKL are
#       the fastest in our experience.  
# 2.  Set NETLIB_DIR, ATLAS_DIR, OPENBLAS_DIR or MKL_DIR to the directory 
#       for your BLAS installation. To find the directory, you can consult
#       the output of:
#           "python -c 'import numpy.distutils.system_info as sysinfo; print([sysinfo.get_info(s) for s in ["blas", "atlas", "openblas", "mkl"]])'"
#       If this returns blank, try to search for the following files on your 
#         system: libblas.so, libatlas.so, libopenblas.so and libmkl_rt.so
#       For Netlib, ATLAS or OpenBLAS, you need to specify the path to 
#         the directory containing corresponding lib*.so file, i.e.: 
#         libblas.so, libatlas.so or libopenblas.so, which usually end with
#         "lib/libblas", "lib/atlas-base" or "lib/openblas-base". 
#       For MKL, you need to specify the path to the directory 
#         containing the "lib" and "include" subdirectories, which usually 
#         ends with "mkl". 
# 3a. If binary executable is desired:
#       Run "make c". 
# 3b. If Python interface is desired:
#       First make sure running "python" invokes the Python installation 
#         that you intend to use. In particular, if you intend to use
#         Python 3, sometimes you need to invoke it using "python3". In 
#         this case, make sure to replace all invocations of "python" 
#         with "python3". 
#       Set PYTHON_DIR to the output of:
#           "python -c 'from distutils.sysconfig import get_python_inc; print(get_python_inc())'" 
#         and NUMPY_DIR to the output of:
#           "python -c 'import numpy as np; print(np.get_include())'". 
#       Run "make py". 
#       If the compiler cannot find Python.h, it means the Python 
#       development headers are not installed. If you believe they are 
#       installed, make sure PYTHON_DIR corresponds to the Python 
#       installation that you intend to use. (Sometimes there are
#       multiple Python installations on your system.)
# 3c. If TensorFlow op is desired:
#       First make sure running "python" invokes the Python installation 
#         that you intend to use. In particular, if you intend to use
#         Python 3, sometimes you need to invoke it using "python3". In 
#         this case, make sure to replace all invocations of "python" 
#         with "python3". 
#       Set TENSORFLOW_LIB_DIR to the output of:
#           "python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())'"
#         and TENSORFLOW_INCL_DIR to the output of:
#           "python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())'". 
#       Run "make tf". 
#     If the compiler complains about not being able to find 
#     libtensorflow.so or libtensorflow_framework.so, download the 
#     libtensorflow*.tar.gz file for your platform and version of 
#     TensorFlow from https://www.tensorflow.org/install/lang_c
#     and copy libtensorflow.so and libtensorflow_framework.so to 
#     TENSORFLOW_LIB_DIR. 

################################################################################
#                                                                              #
# Modify the lines below according to the instructions above. You do not need  #
# to modify PYTHON_DIR and NUMPY_DIR if you do not need the Python interface.  #
# You do not need to modify TENSORFLOW_LIB_DIR and TENSORFLOW_INCL_DIR if you  #
# do not need the TensorFlow op.                                               #
#                                                                              #
################################################################################

# Path to C compiler (no need to modify in most cases)
CC=gcc

# Path to C++ compiler (no need to modify in most cases)
CPPC=g++

# BLAS library name ("netlib", "atlas", "openblas" or "mkl") 
BLAS=netlib

# Path to the directory containing BLAS reference implementation from Netlib, 
# which should contain libblas.so
NETLIB_DIR=/usr/lib/libblas

# Path to ATLAS directory, which should contain libatlas.so
ATLAS_DIR=/usr/lib/atlas-base

# Path to OpenBLAS directory, which should contain libopenblas.so
OPENBLAS_DIR=/usr/lib/openblas-base

# Path to MKL directory, which should contain libmkl_rt.so under the 
# "lib/intel64" subdirectory
MKL_DIR=/opt/intel/mkl

# Path to Python headers directory, which should contain Python.h
PYTHON_DIR=/usr/include/python3.5
#PYTHON_DIR=/usr/include/python2.7

# Path to NumPy headers directory, which should contain arrayobject.h
# under the "numpy" subdirectory
NUMPY_DIR=/usr/local/lib/python3.5/dist-packages/numpy/core/include
#NUMPY_DIR=/usr/local/lib/python2.7/dist-packages/numpy/core/include

# Path to TensorFlow directory, which should contain 
# libtensorflow.so and libtensorflow_framework.so
TENSORFLOW_LIB_DIR=/usr/local/lib/python3.5/dist-packages/tensorflow
#TENSORFLOW_LIB_DIR=/usr/local/lib/python2.7/dist-packages/tensorflow

# Path to TensorFlow headers directory, which should contain tensor.h
# under the "tensorflow/core/framework" subdirectory
TENSORFLOW_INCL_DIR=/usr/local/lib/python3.5/dist-packages/tensorflow/include
#TENSORFLOW_INCL_DIR=/usr/local/lib/python2.7/dist-packages/tensorflow/include

################################################################################
#                                                                              #
# Do not modify the lines below unless you know what you are doing.            #
#                                                                              #
################################################################################

SRC_DIR=src
INCL_DIR=include
PY_SRC_DIR=python
PY_PKG_NAME=dciknn
BUILD_DIR=build
C_BUILD_DIR=$(BUILD_DIR)/c
PY_BUILD_DIR=$(BUILD_DIR)/py
TF_BUILD_DIR=$(BUILD_DIR)/tf

GEN_FLAGS=-Wall -O3 -std=gnu99 -m64 -fopenmp -flto
LIB_FLAGS=-lm

TF_GEN_FLAGS=-Wall -O3 -std=c++11 -m64 -fopenmp -flto -D_GLIBCXX_USE_CXX11_ABI=0
TF_LIB_FLAGS=-L$(TENSORFLOW_LIB_DIR) -ltensorflow_framework -lpthread -ldl -ltensorflow -fPIC

OBJ_FILES=dci.o util.o
INCL_FILES=dci.h util.h

C_OBJ_FILES=$(OBJ_FILES)
TF_OBJ_FILES=$(OBJ_FILES)
PY_OBJ_FILES=py_dci.o $(OBJ_FILES)

ALL_INCL_DIRS=$(INCL_DIR)

ifeq ($(BLAS), netlib)
    LIB_FLAGS += -L$(NETLIB_DIR) -Wl,-rpath $(NETLIB_DIR) -lblas
endif
ifeq ($(BLAS), atlas)
    LIB_FLAGS += -L$(ATLAS_DIR) -Wl,-rpath $(ATLAS_DIR) -latlas
endif
ifeq ($(BLAS), openblas)
    LIB_FLAGS += -L$(OPENBLAS_DIR) -Wl,-rpath $(OPENBLAS_DIR) -lopenblas
endif
ifeq ($(BLAS), mkl)
    ALL_INCL_DIRS += $(MKL_DIR)/include
    GEN_FLAGS += -DUSE_MKL
    LIB_FLAGS += -L$(MKL_DIR)/lib/intel64 -Wl,-rpath $(MKL_DIR)/lib/intel64 -lmkl_rt -lpthread -ldl
endif

C_OBJ_PATHS=$(addprefix $(C_BUILD_DIR)/,$(C_OBJ_FILES))
C_INCL_PATHS=$(addprefix $(INCL_DIR)/,$(INCL_FILES))
C_ALL_INCL_DIRS=$(ALL_INCL_DIRS)
C_ALL_INCL_FLAGS=$(addprefix -I,$(C_ALL_INCL_DIRS))

TF_OBJ_PATHS=$(addprefix $(TF_BUILD_DIR)/,$(TF_OBJ_FILES))
TF_INCL_PATHS=$(addprefix $(INCL_DIR)/,$(INCL_FILES))
TF_ALL_INCL_DIRS=$(ALL_INCL_DIRS) $(TENSORFLOW_INCL_DIR)
TF_ALL_INCL_FLAGS=$(addprefix -I,$(TF_ALL_INCL_DIRS))

PY_OBJ_PATHS=$(addprefix $(PY_BUILD_DIR)/,$(PY_OBJ_FILES))
PY_INCL_PATHS=$(addprefix $(INCL_DIR)/,$(INCL_FILES))
PY_ALL_INCL_DIRS=$(PYTHON_DIR) $(NUMPY_DIR) $(ALL_INCL_DIRS)
PY_ALL_INCL_FLAGS=$(addprefix -I,$(PY_ALL_INCL_DIRS))

.PHONY: all

all: c tf py

.PHONY: c

c: $(C_BUILD_DIR)/example
	ln -sf $(C_BUILD_DIR)/example .

$(C_BUILD_DIR)/example: $(C_BUILD_DIR)/example.o $(C_OBJ_PATHS)
	$(CC) -o $@ $^ $(GEN_FLAGS) $(LIB_FLAGS)

$(C_BUILD_DIR)/%.o: $(SRC_DIR)/%.c $(C_INCL_PATHS)
	mkdir -p $(C_BUILD_DIR)
	$(CC) -c -o $@ $< $(GEN_FLAGS) $(C_ALL_INCL_FLAGS)

.PHONY: tf

tf: $(TF_BUILD_DIR)/_dci_tf.so
	ln -sf $(TF_BUILD_DIR)/_dci_tf.so .

$(TF_BUILD_DIR)/%_tf.so: $(SRC_DIR)/tf%.cc $(TF_OBJ_PATHS)
	$(CPPC) -shared -o $@ $^ -fPIC $(LIB_FLAGS) $(TF_ALL_INCL_FLAGS) $(TF_GEN_FLAGS) $(TF_LIB_FLAGS)

$(TF_BUILD_DIR)/%.o: $(SRC_DIR)/%.c $(TF_INCL_PATHS)
	mkdir -p $(TF_BUILD_DIR)
	$(CC) -c -o $@ $< -fPIC $(GEN_FLAGS) $(TF_ALL_INCL_FLAGS)

.PHONY: py

py: $(PY_BUILD_DIR)/_dci.so
	cp -r $(PY_SRC_DIR)/$(PY_PKG_NAME) .
	ln -sf ../$(PY_BUILD_DIR)/_dci.so $(PY_PKG_NAME)/

$(PY_BUILD_DIR)/%.so: $(PY_BUILD_DIR)/py%.o $(PY_OBJ_PATHS)
	$(CC) -shared -o $@ $^ -fPIC $(GEN_FLAGS) $(LIB_FLAGS)

$(PY_BUILD_DIR)/%.o: $(SRC_DIR)/%.c $(PY_INCL_PATHS)
	mkdir -p $(PY_BUILD_DIR)
	$(CC) -c -o $@ $< -fPIC $(GEN_FLAGS) $(PY_ALL_INCL_FLAGS)

.PHONY: clean

clean: clean-c clean-tf clean-py
	rm -rf $(BUILD_DIR)

.PHONY: clean-c

clean-c:
	rm -rf $(C_BUILD_DIR) example

.PHONY: clean-tf

clean-tf:
	rm -rf $(TF_BUILD_DIR) _dci_tf.so

.PHONY: clean-py

clean-py:
	rm -rf $(PY_BUILD_DIR) $(PY_PKG_NAME)

