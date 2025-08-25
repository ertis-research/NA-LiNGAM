########################################################
################## IMPORT LIBRARIES ####################
########################################################

from utils.environment import GraphEnvLiNGAM
from utils.generate_real_data_graph import RealDataSachs
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
            

def test_algorithm(initial_graph, n_noise, n_tests=10):
    hits_mean = 0
    accuracy_mean = 0
    hits_slow_mean = 0
    accuracy_slow_mean = 0

    for _ in range(n_tests):
        dataset = RealDataSachs(n_noise=n_noise)
        df = dataset.get_dataframe()
        env = GraphEnvLiNGAM(df, initial_graph)

        real_graph = dataset.get_graph()
        real_nodes = list(real_graph.nodes)

        # Get the state of the environment using only real nodes
        optimal_state = np.zeros(len(env.new_features))
        for node in real_nodes:
            if node in env.new_features:
                optimal_state[env.new_features.index(node)] = 1

        calculated_state, _ = env.get_best_state_fast(30)
        calculated_state_slow, _ = env.get_best_state(30)
        print(f"Calculated state: {calculated_state}"
              f"\nOptimal state: {optimal_state}")
        print(f"Calculated state slow: {calculated_state_slow}"
              f"\nOptimal state: {optimal_state}")
        accuracy, hits = check_state(calculated_state, optimal_state)
        accuracy_slow, hits_slow = check_state(calculated_state_slow, optimal_state)
        
        accuracy_mean += accuracy
        hits_mean += hits
        accuracy_slow_mean += accuracy_slow
        hits_slow_mean += hits_slow

    return accuracy_mean / n_tests, hits_mean / n_tests, accuracy_slow_mean / n_tests, hits_slow_mean / n_tests

if __name__ == '__main__':
    accuracy_history = []
    hits_history = []
    accuracy_slow_history = []
    hits_slow_history = []

    max_n_noise = 20
    initial_variables = ['PKC', 'PKA', 'Akt']

    initial_graph = nx.DiGraph()
    initial_graph.add_nodes_from(initial_variables)
    initial_graph.add_edges_from([
        ('PKC', 'PKA'),
        ('PKA', 'Akt')
    ])

    for n_noise in range(max_n_noise + 1):
        print(f'Testing with {n_noise} noise nodes')

        accuracy_mean, hits_mean, accuracy_slow_mean, hits_slow_mean = test_algorithm(initial_graph, n_noise=n_noise, n_tests=20)

        accuracy_history.append(accuracy_mean)
        hits_history.append(hits_mean)
        accuracy_slow_history.append(accuracy_slow_mean)
        hits_slow_history.append(hits_slow_mean)

        model_dict = { 'accuracy': accuracy_history, 'hits': hits_history, 'accuracy_slow': accuracy_slow_history, 'hits_slow': hits_slow_history }

        # Save the model_dict to a file
        with open('graph_metrics/real_accuracy_hits_lingam_slow_20_20.json', 'w') as f:
            json.dump(model_dict, f)

    print('Finished testing')