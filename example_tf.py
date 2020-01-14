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
import tensorflow as tf
import os
import sys

from time import time

def construct_graph(dim, num_comp_indices = 2, num_simp_indices = 7, num_levels = 2, construction_prop_to_visit = 1.0, construction_prop_to_retrieve = 0.002, construction_field_of_view = 10, query_prop_to_visit = 1.0, query_prop_to_retrieve = 0.8, query_field_of_view = 100):
    dci_module = tf.load_op_library('./_dci_tf.so')
    graph = tf.Graph()
    with graph.as_default():
        query = tf.placeholder(tf.float64, shape = [None, None], name = "query")
        data = tf.placeholder(tf.float64, shape = [None, None], name = "data")
        num_neighbours = tf.placeholder(tf.int32, shape = [], name = "num_neighbours")
        update_db = tf.placeholder(tf.bool, shape = [], name = "update_db")
        
        # DCI TensorFlow Op
        # 
        # The op takes in the following attributes, whose values should be specified at graph construction time:
        # 
        # dim:                              Dimensionality of the vectors. 
        # num_comp_indices:                 Number of composite indices (a small integer like 2 or 3 is recommended). 
        # num_simp_indices:                 Number of simple indices per composite index (a larger integer like 7 or 10 is recommended). 
        # num_levels:                       Number of levels (a small integer like 2 or 3 is recommended). 
        # 
        # The op takes in the following input tensors, whose values may remain unknown until runtime:
        # 
        # data:                             A float64 matrix of shape (num of data points) x dim containing the database of points to search over. 
        # query:                            A float32 matrix of shape (num of queries) x dim containing the queries to the database. 
        # num_neighbours:                   An int32 scalar containing the number of nearest neighbours to return. 
        # update_db:                        A boolean scalar specifying whether or not to update the database. If true, will update the database 
        #                                   with the current value of data tensor; if false, will re-use the stale database from before. Must be 
        #                                   set to true the first time this op is run. 
        # construction_prop_to_visit:       A float64 scalar containing the maximum proportion of points to visit when constructing the data 
        #                                   structure. Has no effect when num_levels = 1. A large number like 1.0 is recommended. 
        # construction_prop_to_retrieve:    A float64 scalar containing the maximum proportion of points to retrieve when constructing the data 
        #                                   structure. Has no effect when num_levels = 1. A small number like 0.002 is recommended. 
        # construction_field_of_view:       An int32 scalar containing the maximum number of probes into the next level when constructing the 
        #                                   data structure. Has no effect when num_levels = 1. A moderately large number like 10 is recommended. 
        # query_prop_to_visit:              A float64 scalar containing the maximum proportion of points to visit when querying. A large number 
        #                                   like 1.0 is recommended.
        # query_prop_to_retrieve:           A float64 scalar containing the maximum proportion of points to retrieve when constructing the data 
        #                                   structure. A moderately large number like 0.8 is recommended. 
        # query_field_of_view:              An int32 scalar containing the maximum number of probes into the next level when querying. Has no 
        #                                   effect when num_levels = 1. A large number like 100 is recommended. 
        # 
        # The op returns the following output tensors:
        # 
        # nearest_neighbour_ids:            A int32 matrix of shape (num of queries) x (num of neighbours) containing the indices of the nearest 
        #                                   neighbours to each query. 
        # nearest_neighbour_dists:          A float64 matrix of shape (num of queries) x (num of neighbours) containing the Euclidean distances 
        #                                   between the nearest neighbours and the queries. 
        
        nearest_neighbour_ids, nearest_neighbour_dists = dci_module.dci_knn(data, query, num_neighbours, update_db, dim = dim, num_comp_indices = num_comp_indices, num_simp_indices = num_simp_indices, num_levels = num_levels, construction_prop_to_visit = construction_prop_to_visit, construction_prop_to_retrieve = construction_prop_to_retrieve, construction_field_of_view = construction_field_of_view, query_prop_to_visit = query_prop_to_visit, query_prop_to_retrieve = query_prop_to_retrieve, query_field_of_view = query_field_of_view)
    placeholders = {"query": query, "data": data, "num_neighbours": num_neighbours, "update_db": update_db}
    outputs = {"nearest_neighbour_ids": nearest_neighbour_ids, "nearest_neighbour_dists": nearest_neighbour_dists}
    return graph, placeholders, outputs
    
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
    data_and_queries = gen_data(dim, intrinsic_dim, num_points + num_queries)
    data = np.copy(data_and_queries[:num_points,:])
    queries = data_and_queries[num_points:,:]
    print("Took %.4fs" % (time() - t0))
    
    print("Constructing Graph... ")
    t0 = time()
    graph, placeholders, outputs = construct_graph(dim, num_comp_indices, num_simp_indices, num_levels = num_levels, construction_prop_to_retrieve = construction_prop_to_retrieve, construction_field_of_view = construction_field_of_view, query_prop_to_retrieve = query_prop_to_retrieve, query_field_of_view = query_field_of_view)
    print("Took %.4fs" % (time() - t0))

    print("Starting Tensorflow Session... ")
    t0 = time()
    with tf.Session(graph=graph) as sess:
        print("Took %.4fs" % (time() - t0))
        print("Constructing Data Structure and Querying Using Tensorflow... ")
        t0 = time()
        nearest_neighbour_ids, nearest_neighbour_dists = sess.run([outputs["nearest_neighbour_ids"], outputs["nearest_neighbour_dists"]], feed_dict={placeholders["data"]: data, placeholders["query"]: queries, placeholders["num_neighbours"]: num_neighbours, placeholders["update_db"]: True})
        print("Took %.4fs" % (time() - t0))
        
    print(nearest_neighbour_ids)
    #print(nearest_neighbour_dists)
    
if __name__ == '__main__':
    main(*sys.argv[1:])
