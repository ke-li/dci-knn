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

#ifndef DCI_H
#define DCI_H

// If this is a C++ compiler, use C linkage
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

typedef struct idx_elem {
    double key;
    int local_value;
    int global_value;
} idx_elem;

typedef struct range {
    int start;
    int num;
} range;

typedef struct dci {
    int dim;                        // (Ambient) dimensionality of data
    int num_comp_indices;           // Number of composite indices
    int num_simp_indices;           // Number of simple indices in each composite index
    int num_points;
    int num_levels;
    int num_coarse_points;
    idx_elem** indices;
    double* proj_vec;               // Assuming column-major layout, matrix of size dim x (num_comp_indices*num_simp_indices)
    const double* data;
    range** next_level_ranges;
    int** num_finest_level_points;
} dci;

// Setting num_to_retrieve and prop_to_retrieve has no effect when blind is true
// Setting field_of_view has no effect when there is only one level
// min_num_finest_level_points is for internal use only; setting it has no effect (since it will be overwritten)
typedef struct dci_query_config {
    bool blind;
    // Querying algorithm terminates whenever we have visited max(num_visited, prop_visited*num_points) points or retrieved max(num_retrieved, prop_retrieved*num_points) points, whichever happens first
    int num_to_visit;
    int num_to_retrieve;
    double prop_to_visit;
    double prop_to_retrieve;
    int field_of_view;
    int min_num_finest_level_points;
} dci_query_config;

void dci_init(dci* const dci_inst, const int dim, const int num_comp_indices, const int num_simp_indices);

// Note: the data itself is not kept in the index and must be kept in-place
void dci_add(dci* const dci_inst, const int dim, const int num_points, const double* const data, const int num_levels, const dci_query_config construction_query_config);

// CAUTION: This function allocates memory for each nearest_neighbours[j], nearest_neighbour_dists[j], so we need to deallocate them outside of this function!
void dci_query(dci* const dci_inst, const int dim, const int num_queries, const double* const query, const int num_neighbours, const dci_query_config query_config, int** const nearest_neighbours, double** const nearest_neighbour_dists, int* const num_returned);

void dci_clear(dci* const dci_inst);

// Clear indices and reset the projection directions
void dci_reset(dci* const dci_inst);

void dci_free(const dci* const dci_inst);

#ifdef __cplusplus
}
#endif

#endif // DCI_H
