####################################
############# LIBRARIES ############
####################################

from utils.nalingam_environment import GraphEnvNALiNGAM

from utils.generate_real_data_graph import RealDataSachs
from utils.generate_synthetic_graph import SyntheticGraphGenerator
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, PowerTransformer

import seaborn as sns
import itertools

import numpy as np
import pandas as pd
import os
import json

import networkx as nx

import concurrent.futures

from causallearn.utils.Dataset import load_dataset

from networkx.readwrite import json_graph
import json

from utils.cdmodels.pc import PCAlgorithm
from utils.cdmodels.fci import FCIAlgorithm
from utils.cdmodels.ges import GESAlgorithm
from utils.cdmodels.gin import GINAlgorithm
from utils.cdmodels.exact_search import ExactSearchAlgorithm
from utils.cdmodels.grasp import GRaSPAlgorithm
from utils.cdmodels.granger import GrangerAlgorithm
from utils.nalingam_score import NALiNGAMScoreAlgorithm

from cdt.metrics import precision_recall, SID, SHD

import warnings
warnings.filterwarnings("ignore")

#############################################
############## FUNCTIONS ####################
#############################################

def get_initial_subgraph(graph, initial_variables):
    """
    Returns a subgraph of the initial graph containing only the initial variables.
    """
    subgraph = graph.subgraph(initial_variables).copy()
    return subgraph

def save_graph_to_json(graph, n_noise, iterations, folder):
    data = json_graph.node_link_data(graph)
    with open(f'{folder}/graph_noise_{str(n_noise)}_it_{str(iterations)}.json', 'w') as f:
        json.dump(data, f)

def load_graph_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return json_graph.node_link_graph(data)

def is_path(graph, start, end):
    visited = set()
    stack = [start]

    while stack:
        node = stack.pop()
        if node == end:
            return True
        if node not in visited:
            if graph.has_node(node):
                visited.add(node)
                stack.extend(graph.neighbors(node))

    return False

def check_graph(graph, real_graph):
    real_edges_accuracy = 0
    invented_edges_accuracy = 0
    invented_wrong_edges_accuracy = 0

    for edge in list(real_graph.edges()):
        if edge in list(graph.edges()):
            real_edges_accuracy += 1
    if real_edges_accuracy > 0:
        real_edges_accuracy = real_edges_accuracy / len(list(real_graph.edges()))

    for edge in list(graph.edges()):
        if edge not in list(real_graph.edges()):
            invented_edges_accuracy += 1
    if invented_edges_accuracy > 0:
        invented_edges_accuracy = invented_edges_accuracy / len(list(graph.edges()))

    for edge in list(graph.edges()):
            if not is_path(real_graph, edge[0], edge[1]):
                invented_wrong_edges_accuracy += 1
    if invented_wrong_edges_accuracy > 0:
        invented_wrong_edges_accuracy = invented_wrong_edges_accuracy / len(list(graph.edges()))

    return real_edges_accuracy, invented_edges_accuracy, invented_wrong_edges_accuracy

def test_real_graph_generation(initial_graph, functions, max_n_noise=20, iterations=20, folder='test'):

    for n_noise in range(max_n_noise + 1):
        print(f"Generating dataset with {n_noise} noise variables")
        
        for i in range(iterations):
            dataset = RealDataSachs(n_noise=n_noise)
            df = dataset.get_dataframe()
            
            if n_noise == 0 and i == 0:
                # Save the real graph only once
                path = f'results_real/{folder}/real_graph'
                if not os.path.exists(path):
                    os.makedirs(path)
                save_graph_to_json(dataset.get_graph(), n_noise, i+1, folder=path)

            if (i+1) % 10 == 0:
                print(f"Iteration {i+1}/{iterations}")
            for discovery_method in functions:
                print(f"Running discovery method: {discovery_method}")
                if discovery_method == 'NALiNGAMAlgorithm':
                    env = GraphEnvNALiNGAM(df, initial_graph)
                    found_state, _ = env.get_best_state_fast(30)

                    graph = env.get_graph(found_state, iterations=20)
                elif discovery_method == 'LiNGAMAlgorithm':
                    env = GraphEnvNALiNGAM(df, initial_graph)

                    graph = env.get_graph(np.ones(len(df.columns) - len(list(initial_graph.nodes()))), iterations=1) # Using NA-LiNGAM with 1 iteration to simulate LiNGAM without score
                else:
                    try:
                        discovery_method_function = globals()[discovery_method]
                        graph = discovery_method_function(df, df.columns).get_graph()
                    except KeyError:
                        print(f"Discovery method {discovery_method} not found.")
                        continue
                path = f'results_real/{folder}/{discovery_method}'
                # If folder does not exist, create it
                if not os.path.exists(path):
                    os.makedirs(path)
                save_graph_to_json(graph, n_noise, i+1, folder=path)

def test_synthetic_graph_generation(starting_nodes, functions, dataset_size=1000, n_nodes=20, iterations=100, folder='test'):

    for n_noise in range(n_nodes - starting_nodes + 1):
        print(f"Generating dataset with {n_noise} noise variables")

        dataset = SyntheticGraphGenerator(
            sample_size=dataset_size,
            node_size=n_nodes,
            min_edges=2,
            max_edges=5,
            n_noise_nodes=n_noise
        )

        for i in range(iterations):
            dataset.generate_samples()

            if (i+1) % 10 == 0:
                    print(f"Iteration {i+1}/{iterations}")

            real_variables, _ = dataset.get_real_noise_nodes()
            initial_variables = real_variables[:starting_nodes]
            initial_graph = get_initial_subgraph(dataset.get_graph(), initial_variables)

            path = f'results_synthetic/{folder}/real_graph'
            if not os.path.exists(path):
                os.makedirs(path)
            save_graph_to_json(dataset.get_graph(), n_noise, i+1, folder=path)

            df = dataset.get_dataframe()

            for discovery_method in functions:
                print(f"Running discovery method: {discovery_method}")
                if discovery_method == 'NALiNGAMAlgorithm':
                    env = GraphEnvNALiNGAM(df, initial_graph)
                    found_state, _ = env.get_best_state_fast(30)

                    graph = env.get_graph(found_state, iterations=20)
                elif discovery_method == 'LiNGAMAlgorithm':
                    env = GraphEnvNALiNGAM(df, initial_graph)

                    graph = env.get_graph(np.ones(len(df.columns) - len(list(initial_graph.nodes()))), iterations=1)
                else:
                    try:
                        discovery_method_function = globals()[discovery_method]
                        graph = discovery_method_function(df, df.columns).get_graph()
                    except KeyError:
                        print(f"Discovery method {discovery_method} not found.")
                        continue
                path = f'results_synthetic/{folder}/{discovery_method}'
                # If folder does not exist, create it
                if not os.path.exists(path):
                    os.makedirs(path)
                save_graph_to_json(graph, n_noise, i+1, folder=path)

def evaluate_graphs(graphs_dict, real_graphs, is_real_data):
    results = {}

    if is_real_data:
        real_graph = real_graphs['real']

    for method in graphs_dict.keys():
        results[method] = {}

    for method, graph_info in graphs_dict.items():
        method_results = {'AUC': [], 'SID': [], 'SHD': [], 'Edges Accuracy': [],
                         'Invented Edges Accuracy': [], 'Invented Wrong Edges Accuracy': []}
        for noise_level, graphs in graph_info.items():
            print(f"Evaluating method: {method}, Noise level: {noise_level}")

            temp_auc = 0
            temp_sid = 0
            temp_shd = 0
            edges_accuracy = 0
            invented_edges_accuracy = 0
            invented_wrong_edges_accuracy = 0
            for i, graph in enumerate(graphs):
                if not is_real_data:
                    real_graph = real_graphs[f'{noise_level}_{i}']
                temp_edges_accuracy, temp_invented_edges_accuracy, temp_invented_wrong_edges_accuracy = check_graph(graph, real_graph)
                edges_accuracy += temp_edges_accuracy
                invented_edges_accuracy += temp_invented_edges_accuracy
                invented_wrong_edges_accuracy += temp_invented_wrong_edges_accuracy

                # Add nodes from real_graph to the graph if they are missing
                missing_nodes = set(real_graph.nodes()) - set(graph.nodes())
                for node in missing_nodes:
                    graph.add_node(node)
                predicted_scores = [metric(real_graph, graph) for metric in (precision_recall, SID, SHD)]
                temp_auc += predicted_scores[0][0]
                temp_sid += predicted_scores[1]
                temp_shd += predicted_scores[2]

            num_graphs = len(graphs)
            if num_graphs > 0:
                method_results['AUC'].append(temp_auc / num_graphs)
                method_results['SID'].append(temp_sid / num_graphs)
                method_results['SHD'].append(temp_shd / num_graphs)
                method_results['Edges Accuracy'].append(edges_accuracy / num_graphs)
                method_results['Invented Edges Accuracy'].append(invented_edges_accuracy / num_graphs)
                method_results['Invented Wrong Edges Accuracy'].append(invented_wrong_edges_accuracy / num_graphs)

            results[method] = method_results.copy()

    return results