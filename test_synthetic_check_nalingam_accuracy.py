########################################################
################## IMPORT LIBRARIES ####################
########################################################

from utils.environment import GraphEnvLiNGAM
from utils.generate_synthetic_graph import SyntheticGraphGenerator
from utils.graph_functions import get_initial_subgraph

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx

import seaborn as sns
import json

def check_state(state, real_state):
    optimal_state_count = 0
    variable_accuracy = 0
    for i in range(len(state)):
        if state[i] == real_state[i]:
            variable_accuracy += 1
    
    if variable_accuracy > 0:
        variable_accuracy = variable_accuracy / len(state)
    if variable_accuracy == 1:
        optimal_state_count = 1

    return variable_accuracy, optimal_state_count
            

def test_algorithm(syn_graph, n_tests=10, starting_nodes=3):
    hits_mean = 0
    accuracy_mean = 0
    hits_slow_mean = 0
    accuracy_slow_mean = 0

    for _ in range(n_tests):
        syn_graph.generate_samples()

        real_nodes, _ = syn_graph.get_real_noise_nodes()
        initial_variables = real_nodes[:starting_nodes]

        dataset = syn_graph.get_dataframe()

        real_variables, _ = syn_graph.get_real_noise_nodes()
        initial_variables = real_variables[:starting_nodes]
        initial_graph = get_initial_subgraph(syn_graph.get_graph(), initial_variables)

        env = GraphEnvLiNGAM(dataset, initial_graph)

        # Get the state of the environment using only real nodes
        optimal_state = np.zeros(len(env.new_features))
        for node in real_nodes:
            if node in env.new_features:
                optimal_state[env.new_features.index(node)] = 1

        calculated_state, _ = env.get_best_state_fast()
        calculated_state_slow, _ = env.get_best_state()

        accuracy, hits = check_state(calculated_state, optimal_state)
        accuracy_slow, hits_slow = check_state(calculated_state_slow, optimal_state)
        
        accuracy_mean += accuracy
        hits_mean += hits
        accuracy_slow_mean += accuracy_slow
        hits_slow_mean += hits_slow

    return accuracy_mean / n_tests, hits_mean / n_tests, accuracy_slow_mean / n_tests, hits_slow_mean / n_tests

if __name__ == '__main__':
    dataset_size = 1000
    n_tests = 1
    nodes = 20
    starting_nodes = 3

    accuracy_history = []
    hits_history = []
    accuracy_slow_history = []
    hits_slow_history = []

    for i in range(1, nodes - starting_nodes + 1):
        print(f'Testing with {i} nodes')
        syn_graph = SyntheticGraphGenerator(
            sample_size=dataset_size,
            node_size=i+starting_nodes,
            min_edges=2,
            max_edges=5,
            n_noise_nodes=int(i/2)
        )

        accuracy_mean, hits_mean, accuracy_slow_mean, hits_slow_mean = test_algorithm(syn_graph, n_tests, starting_nodes)

        accuracy_history.append(accuracy_mean)
        hits_history.append(hits_mean)
        accuracy_slow_history.append(accuracy_slow_mean)
        hits_slow_history.append(hits_slow_mean)

        model_dict = { 'accuracy': accuracy_history, 'hits': hits_history, 'accuracy_slow': accuracy_slow_history, 'hits_slow': hits_slow_history }

        # Save the model_dict to a file
        with open('graph_metrics/synthetic_accuracy_hits_lingam_slow_100_20.json', 'w') as f:
            json.dump(model_dict, f)

    print('Finished testing')