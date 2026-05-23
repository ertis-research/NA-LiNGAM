########################################################
################## IMPORT LIBRARIES ####################
########################################################

import sys
sys.path.append('../')

from utils.nalingam_environment import GraphEnvNALiNGAM
from utils.generate_synthetic_graph import SyntheticGraphGenerator
from utils.graph_functions import get_initial_subgraph

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx

import seaborn as sns
import json
import time

def test_algorithm(syn_graph, n_tests=10, starting_nodes=2):
    """
    Test the execution time of the optimized NALiNGAM algorithm.

    Parameters:
    - syn_graph (SyntheticGraphGenerator): The synthetic graph generator instance.
    - n_tests (int): Number of tests to run for averaging execution time.
    - starting_nodes (int): Number of initial nodes in the subgraph.

    Returns:
    - fast_time_mean (float): Average execution time of the optimized NALiNGAM
    """
    fast_time_mean = 0

    for _ in range(n_tests):
        syn_graph.generate_samples()

        real_variables, _ = syn_graph.get_real_noise_nodes()
        initial_variables = real_variables[:starting_nodes]
        initial_graph = get_initial_subgraph(syn_graph.get_graph(), initial_variables)

        dataset = syn_graph.get_dataframe()
        env = GraphEnvNALiNGAM(dataset, initial_graph)

        # Get the state of the environment using only real variables
        optimal_state = np.zeros(len(env.new_features))
        for node in real_variables:
            if node in env.new_features:
                optimal_state[env.new_features.index(node)] = 1

        fast_time_start = time.time()
        env.get_best_state_fast()
        fast_time_end = time.time()

        fast_time_mean += fast_time_end - fast_time_start

    return fast_time_mean / n_tests

if __name__ == '__main__':
    dataset_size = 1000
    n_tests = 3
    min_nodes = 0
    max_nodes = 100
    initial_starting_nodes = 3

    fast_algorithm_time_history = []

    for i in range(100):
        starting_nodes = initial_starting_nodes + i
        additional_nodes = int(starting_nodes / 2)

        print(f'Testing with {starting_nodes} + {additional_nodes} variables')
        syn_graph = SyntheticGraphGenerator(
            sample_size=dataset_size,
            node_size=starting_nodes + additional_nodes,
            min_edges=2,
            max_edges=5,
            n_noise_nodes=0
        )

        fast_algorithm_time_mean = test_algorithm(syn_graph, n_tests, starting_nodes)

        fast_algorithm_time_history.append(fast_algorithm_time_mean)

        model_dict = { 'fast_time': fast_algorithm_time_history }

        # Save the model_dict to a file
        with open('graph_metrics/execution_time_fast_addition.json', 'w') as f:
            json.dump(model_dict, f)

    print('Finished testing')