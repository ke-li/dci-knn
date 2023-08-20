/*
 * Code for Fast k-Nearest Neighbour Search via Prioritized DCI
 * 
 * This code implements the method described in the Prioritized DCI paper, 
 * which can be found at https://arxiv.org/abs/1703.00440
 * 
 * This file is a part of the Dynamic Continuous Indexing reference 
 * implementation.
 * 
 * 
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. 
 * 
 * Copyright (C) 2017    Ke Li
 */

#ifndef UTIL_H
#define UTIL_H

// If this is a C++ compiler, use C linkage
#ifdef __cplusplus
extern "C" {
#endif

#ifdef USE_MKL
#define DGEMM dgemm
#else
#define DGEMM dgemm_
#endif  // USE_MKL

// BLAS native Fortran interface
extern void DGEMM(const char* const transa, const char* const transb, const int* const m, const int* const n, const int* const k, const double* const alpha, const double* const A, const int* const lda, const double* const B, const int* const ldb, const double* const beta, double* const C, const int* const ldc);

void matmul(const int M, const int N, const int K, const double* const A, const double* const B, double* const C);

void gen_data(double* const data, const int ambient_dim, const int intrinsic_dim, const int num_points);

double compute_dist(const double* const vec1, const double* const vec2, const int dim);

double rand_normal();

void print_matrix(const double* const data, const int num_rows, const int num_cols);

#ifdef __cplusplus
}
#endif

#endif // UTIL_H
