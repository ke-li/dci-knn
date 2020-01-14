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

import numpy as np
import os
import sys
sys.path.append(os.getcwd())

from dciknn import DCI

from time import time

def gen_data(ambient_dim, intrinsic_dim, num_points):
    latent_data = 2 * np.random.rand(num_points, intrinsic_dim) - 1     # Uniformly distributed on [-1,1)
    transformation = 2 * np.random.rand(intrinsic_dim, ambient_dim) - 1
    data = np.dot(latent_data, transformation)
    return data     # num_points x ambient_dim

def main(*args):
    
    #############################################################################################################################################
    #                                                                                                                                           #
    # Data Generation Hyperparameters                                                                                                           #
    #                                                                                                                                           #
    #############################################################################################################################################
    
    dim = 5000              # Dimensionality of data
    intrinsic_dim = 50      # Intrinsic dimensionality of data
    num_points = 10000      # Number of data points
    num_queries = 5         # Number of queries
    
    #############################################################################################################################################
    #                                                                                                                                           #
    # Problem Hyperparameter                                                                                                                    #
    #                                                                                                                                           #
    #############################################################################################################################################
    
    num_neighbours = 10      # The k in k-NN
    
    #############################################################################################################################################
    #                                                                                                                                           #
    # DCI Hyperparameters                                                                                                                       #
    #                                                                                                                                           #
    #############################################################################################################################################
    
    # Guide for tuning hyperparameters:
    
    # num_comp_indices:                 trades off accuracy vs. construction and query time - high values lead to more accurate results, but 
    #                                   slower construction and querying
    # num_simp_indices:                 trades off accuracy vs. construction and query time - high values lead to more accurate results, but 
    #                                   slower construction and querying. If num_simp_indices is increased, may need to increase 
    #                                   num_comp_indices. If the intrisic dimensionality of dataset increases, should increase num_simp_indices 
    #                                   proportionally. 
    # num_levels:                       trades off construction time vs. query time - higher values lead to faster querying, but slower 
    #                                   construction. If num_levels is increased, may need to increase query_field_of_view and 
    #                                   construction_field_of_view. If the number of data points increases substantially, should increase
    #                                   num_levels logarithmically. 
    # construction_field_of_view:       trades off accuracy/query time vs. construction time - higher values lead to *slightly* more accurate 
    #                                   results and/or *slightly* faster querying, but *slightly* slower construction. If the number of data
    #                                   points increases, need to increase construction_field_of_view at a rate of n^(1/num_levels) unless 
    #                                   num_levels is increased. 
    # construction_prop_to_retrieve:    trades off acrruacy vs. construction time - higher values lead to *slightly* more accurate results, 
    #                                   but slower construction. If the number of data points increases, construction_prop_to_retrieve should 
    #                                   remain roughly the same. If the intrisic dimensionality of dataset increases, should increase 
    #                                   construction_prop_to_retrieve slightly.  
    # query_field_of_view:              trades off accuracy vs. query time - higher values lead to more accurate results, but *slightly* slower 
    #                                   querying. If the number of data points increases, need to increase query_field_of_view at a rate of 
    #                                   n^(1/num_levels) unless num_levels is increased. 
    # query_prop_to_retrieve:           trades off accuracy vs. query time - higher values lead to more accurate results, but slower querying. 
    #                                   If the number of data points increases, query_prop_to_retrieve should remain roughly the same. If the 
    #                                   intrisic dimensionality of dataset increases, should increase query_prop_to_retrieve slightly. 
    
    num_comp_indices = 2
    num_simp_indices = 7
    num_levels = 2
    construction_field_of_view = 10
    construction_prop_to_retrieve = 0.002
    query_field_of_view = 100
    query_prop_to_retrieve = 0.8
    
    print("Generating Data... ")
    t0 = time()
    data_and_query = gen_data(dim, intrinsic_dim, num_points + num_queries)
    data = np.copy(data_and_query[:num_points,:])
    query = data_and_query[num_points:,:]
    
    print("Took %.4fs" % (time() - t0))
    
    print("Constructing Data Structure... ")
    t0 = time()
    
    # DCI()
    # 
    # Constructs a new DCI database. 
    # 
    # The constructor takes in the following parameters:
    #
    # dim:                              Dimensionality of the vectors. 
    # num_comp_indices:                 Number of composite indices (a small integer like 2 or 3 is recommended). 
    # num_simp_indices:                 Number of simple indices per composite index (a larger integer like 7 or 10 is recommended). 
    dci_db = DCI(dim, num_comp_indices, num_simp_indices)
    
    
    # DCI.add()
    # 
    # Add data to DCI database. 
    # 
    # The method takes in the following parameters:
    # 
    # data:                             A float64 matrix of shape (num of data points) x dim containing the database of points to search over. 
    # num_levels:                       Number of levels (a small integer like 2 or 3 is recommended). 
    # field_of_view:                    Maximum number of probes into the next level when constructing the data structure. Has no effect when 
    #                                   num_levels = 1. A moderately large number like 10 is recommended. 
    # prop_to_visit:                    Maximum proportion of points to visit when constructing the data structure. Has no effect when 
    #                                   num_levels = 1. A large number like 1.0 is recommended. 
    # prop_to_retrieve:                 Maximum proportion of points to retrieve when constructing the data structure. Has no effect when 
    #                                   num_levels = 1. A small number like 0.002 is recommended. 
    # blind:                            Whether to look at the data and compute true distances between the data points at each level and the 
    #                                   retrieved data points at the next level. When set to true, the association between data points in
    #                                   adjacent levels will be less accurate. To compensate for this, num_simp_indices should be increased. 
    #                                   Has no effect when num_levels = 1. It is recommended to set this to false. 
    dci_db.add(data, num_levels = num_levels, field_of_view = construction_field_of_view, prop_to_retrieve = construction_prop_to_retrieve)
    
    print("Took %.4fs" % (time() - t0))
    
    print("Querying... ")
    t0 = time()
    
    # DCI.query()
    # 
    # Query the DCI database. 
    # 
    # The method takes in the following parameters:
    # 
    # query:                            A float32 matrix of shape (num of queries) x dim containing the queries to the database. 
    # num_neighbours:                   The number of nearest neighbours to return. 
    # field_of_view:                    Maximum number of probes into the next level when querying. Has no effect when num_levels = 1. 
    #                                   A large number like 100 is recommended. 
    # prop_to_visit:                    Maximum proportion of points to visit when querying. A large number like 1.0 is recommended.
    # prop_to_retrieve:                 Maximum proportion of points to retrieve when constructing the data structure. A moderately large 
    #                                   number like 0.8 is recommended. 
    # blind:                            Whether to look at the data and compute true distances between the query and the retrieved points. 
    #                                   When set to true, nearest_neighbour_dists will contain the maximum projected distances rather than
    #                                   true distances and the returned points will be less accurate. To compensate for this, 
    #                                   num_simp_indices should be increased. It is recommended to set this to false. 
    # 
    # The method returns the following:
    # 
    # nearest_neighbour_ids:            A list of int32 arrays containing the indices of the nearest neighbours to each query. 
    # nearest_neighbour_dists:          A list of float64 arrays containing the Euclidean distances between the nearest neighbours and the 
    #                                   queries. 
    nearest_neighbour_idx, nearest_neighbour_dists = dci_db.query(query, num_neighbours = num_neighbours, field_of_view = query_field_of_view, prop_to_retrieve = query_prop_to_retrieve)
    
    print("Took %.4fs" % (time() - t0))
    print(nearest_neighbour_idx)
    print(nearest_neighbour_dists)
    
if __name__ == '__main__':
    main(*sys.argv[1:])
