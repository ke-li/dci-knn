# Fast k-Nearest Neighbour Search using Dynamic Continuous Indexing (DCI)

### [Project](https://people.eecs.berkeley.edu/~ke.li/projects/dci/) | [Slides](https://people.eecs.berkeley.edu/~ke.li/papers/dci_slides.pdf) | [Poster](https://people.eecs.berkeley.edu/~ke.li/papers/pdci_icml17_poster.pdf) | [Talk Video](https://vimeo.com/238229447) | [Paper](https://arxiv.org/abs/1703.00440)

Dynamic Continuous Indexing (DCI) is a family of randomized algorithms for exact _k_-nearest neighbour search that overcomes the curse of dimensionality. Its query time complexity is linear in ambient dimensionality and sublinear in intrinsic dimensionality. Details of the algorithm and analysis of time complexity can be found in the following papers:

"[Fast _k_-Nearest Neighbour Search via Dynamic Continuous Indexing](https://arxiv.org/abs/1512.00442)", _International Conference on Machine Learning (ICML)_, 2016\
"[Fast _k_-Nearest Neighbour Search via Prioritized DCI](https://arxiv.org/abs/1703.00440)", _International Conference on Machine Learning (ICML)_, 2017

This repository contains the reference implementation of Prioritized DCI, which was written in C to take advantage of compile-time optimizations and multi-threading. It comes with a C interface, a Python 2 & 3 interface and a TensorFlow op. Currently, the code only runs on the CPU. GPU support will be added in the future. 

# Prerequisites

1. A C compiler with support for OpenMP, e.g.: gcc
2. Python 2.7+ or Python 3.1+
3. A BLAS library (supported implementations include the reference implementation from Netlib, ATLAS, OpenBLAS and MKL)
4. Python development headers
5. (If Python interface is desired) NumPy
6. (If TensorFlow op is desired) TensorFlow and C++ compiler

# Setup

The library can be compiled in one of two ways: using Python distutils or the good old Makefile. The former requires less manual configuration, but *cannot* be used if your code uses the C interface or the TensorFlow op. 

## Option 1: Python distutils

In the root directory of the code base, go into the "python" subdirectory:
```bash
cd python
```

If you have sudo access, run the following command to compile and install as a Python package:
(If your Python interpreter is named differently, e.g.: "python3", you will need to replace all occurrences of "python" with "python3" in the commands below.)
```bash
python setup.py install
```

If you do not have sudo access, run the following command instead:
```bash
python setup.py install --user
```

## Option 2: Makefile 

In the root directory of the code base, follow the instructions in the Makefile to specify the paths to BLAS, and optionally, Python, NumPy and TensorFlow. 

### Python Interface

If you would like to build the Python interface, run the following from the root directory of the code base:
```bash
make py
```

If you would like to use DCI in a script outside of the root directory of the code base, either add a symbolic link to the "dciknn" subdirectory within the directory containing your script, or add the root directory of the code base to your PYTHONPATH environment variable. 

### TensorFlow Op

If you would like to build the TensorFlow op, run the following from the root directory of the code base:
```bash
make tf
```

### C Interface

If you would like to build a binary executable from code that uses the C interface, run the following from the root directory of the code base:
```bash
make c
```

# Getting Started

In the root directory of the code base, execute the following commands, depending on which interface you would like to use:

### Python Interface

```bash
python example.py
```

### TensorFlow Op

```bash
python example_tf.py
```

### C Interface

```bash
./example
```

See the source code for example usage. The source code of the binary executable that uses the C interface is in "src/example.c".

## Reference

Please cite the following paper if you found this library useful in your research:

### Fast _k_-Nearest Neighbour Search via Dynamic Continuous Indexing
[Ke Li](https://people.eecs.berkeley.edu/~ke.li/), [Jitendra Malik](https://people.eecs.berkeley.edu/~malik/)
*International Conference on Machine Learning (ICML)*, 2016

```
@inproceedings{li2016fast,
  title={Fast k-nearest neighbour search via {Dynamic Continuous Indexing}},
  author={Li, Ke and Malik, Jitendra},
  booktitle={International Conference on Machine Learning},
  pages={671--679},
  year={2016}
}
```
