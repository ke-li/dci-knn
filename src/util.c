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

#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <math.h>
#include "util.h"

// Assuming column-major layout, computes A^T * B. A is K x M, B is K x N, and C is M x N. 
void matmul(const int M, const int N, const int K, const double* const A, const double* const B, double* const C) {
    const char TRANSA = 'T';
    const char TRANSB = 'N';
    const double ALPHA = 1.; 
    const double BETA = 0.; 
    const int LDA = K;
    const int LDB = K;
    const int LDC = M;
    DGEMM(&TRANSA, &TRANSB, &M, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC);
}

void gen_data(double* const data, const int ambient_dim, const int intrinsic_dim, const int num_points) {
    int i;
    double* latent_data = (double *)memalign(64, sizeof(double)*intrinsic_dim*num_points);
    double* transformation = (double *)memalign(64, sizeof(double)*intrinsic_dim*ambient_dim);
    for (i = 0; i < intrinsic_dim*num_points; i++) {
        latent_data[i] = 2 * drand48() - 1;
    }
    for (i = 0; i < intrinsic_dim*ambient_dim; i++) {
        transformation[i] = 2 * drand48() - 1;
    }
    // Assuming column-major layout, transformation is intrisic_dim x ambient_dim, 
    // latent_data is intrinsic_dim x num_points, data is ambient_dim x num_points
    matmul(ambient_dim, num_points, intrinsic_dim, transformation, latent_data, data);
    free(latent_data);
    free(transformation);
}

double compute_dist(const double* const vec1, const double* const vec2, const int dim) {
    int i;
    double sq_dist = 0.0;
    for (i = 0; i < dim; i++) {
        sq_dist += (vec1[i] - vec2[i])*(vec1[i] - vec2[i]);
    }
    return sqrt(sq_dist);
}

double rand_normal() {
    static double V1, V2, S;
    static int phase = 0;
    double X;

    if(phase == 0) {
        do {
            double U1 = drand48();
            double U2 = drand48();
            
            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
            } while(S >= 1 || S == 0);

        X = V1 * sqrt(-2 * log(S) / S);
    } else
        X = V2 * sqrt(-2 * log(S) / S);

    phase = 1 - phase;

    return X;
}

// Print matrix assuming column-major layout
void print_matrix(const double* const data, const int num_rows, const int num_cols) {
    int i, j;
    for (i = 0; i < num_rows; i++) {
        for (j = 0; j < num_cols; j++) {
            printf("%.4f\t", data[i+j*num_rows]);
        }
        printf("\n");
    }
}
