####################################
############# LIBRARIES ############
####################################

from utils.nalingam_environment import GraphEnvNALiNGAM

from utils.generate_real_data_graph import RealDataSachs, AdaByronData
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
from utils.cdmodels.boss import BOSSAlgorithm
from utils.cdmodels.gin import GINAlgorithm
from utils.cdmodels.exact_search import ExactSearchAlgorithm
from utils.cdmodels.grasp import GRaSPAlgorithm
from utils.cdmodels.granger import GrangerAlgorithm
from utils.nalingam_score import NALiNGAMScoreAlgorithm
from utils.cdmodels.notears import NOTEARSAlgorithm
from utils.cdmodels.cam import CAMAlgorithm
from utils.cdmodels.ccdr import CCDrAlgorithm
from utils.cdmodels.gies import GIESAlgorithm

from cdt.metrics import precision_recall, SID, SHD

import warnings
warnings.filterwarnings("ignore")

#############################################
############## FUNCTIONS ####################
#############################################

def get_initial_subgraph(graph, initial_variables):
    """
    Returns the subgraph of the given graph containing only the initial variables.

    Parameters:
    - graph (networkx.DiGraph): The original graph.
    - initial_variables (list): The list of initial variables to include in the subgraph.

    Returns:
    - networkx.DiGraph: The subgraph containing only the initial variables.
    """
    subgraph = nx.DiGraph(nx.subgraph(graph, initial_variables))
    return subgraph

def save_graph_to_json(graph, n_noise, iterations, folder):
    """
    Saves a graph to a json file.
    
    Parameters:
    - graph (networkx.DiGraph): The graph to save.
    - n_noise (int): The number of noise variables to name the file.
    - iterations (int): The iteration number to name the file.
    - folder (str): The folder to save the graph in.
    """
    if graph is not None:
        data = json_graph.node_link_data(graph)
    else:
        data = None
    file_path = os.path.join(folder, f'graph_noise_{n_noise}_it_{iterations}.json')
    with open(file_path, 'w') as f:
        json.dump(data, f)

def load_graph_from_json(file_path):
    """
    Loads a graph from a json file.

    Parameters:
    - file_path (str): The path to the json file.

    Returns:
    - networkx.DiGraph: The loaded graph.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    if data is None:
        return None
    else:
        return json_graph.node_link_graph(data)

def is_path(graph, start, end):
    """
    Check if there is a path from start to end in the directed graph.

    Parameters:
    - graph (networkx.DiGraph): The directed graph.
    - start (node): The starting node.
    - end (node): The ending node.

    Returns:
    - bool: True if there is a path from start to end, False otherwise.
    """
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
    """
    Compares two directed graphs and calculates the accuracy of real edges, invented edges, and invented wrong edges.

    Parameters:
    - graph (networkx.DiGraph): The graph to evaluate.
    - real_graph (networkx.DiGraph): The ground truth graph.

    Returns:
    - tuple: A tuple containing three accuracy metrics:
        - real_edges_accuracy (float): The accuracy of real edges.
        - invented_edges_accuracy (float): The accuracy of invented edges (edges that are not in the real graph).
        - invented_wrong_edges_accuracy (float): The accuracy of invented wrong edges (edges with a wrong direction).
    """
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

def test_real_graph_generation(initial_graph, functions, max_n_noise=20, iterations=20, folder='test', dataset_class=RealDataSachs, nalingam_score_iterations=30, nalingam_graph_iterations=20):
    """
    Tests various graph discovery methods for graph composition on the Sachs real dataset with added noise variables.

    Parameters:
    - initial_graph (networkx.DiGraph): The initial subgraph to start the discovery from.
    - functions (list): A list of graph discovery method names as strings.
    - max_n_noise (int): The maximum number of noise variables to add.
    - iterations (int): The number of iterations to run for each noise level.
    - folder (str): The folder to save the results in.

    Saves the real graph only once when n_noise is 0 and iteration is 0.
    Saves the discovered graphs for each method, noise level and iteration.
    """

    for n_noise in range(0, max_n_noise + 1):
        print(f"Generating dataset with {n_noise} noise variables")
        
        for i in range(iterations):
            dataset = dataset_class(n_noise=n_noise)
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
                try:
                    print(f"Running discovery method: {discovery_method}")
                    if discovery_method == 'NALiNGAMAlgorithm':
                        env = GraphEnvNALiNGAM(df, initial_graph)
                        found_state, _ = env.get_best_state_fast(nalingam_score_iterations)

                        graph = env.get_graph(found_state, iterations=nalingam_graph_iterations)
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
                except Exception as e:
                    print(f"Error running discovery method {discovery_method} with n_noise={n_noise} and iteration={i+1}: {e}")
                    path = f'results_real/{folder}/{discovery_method}'
                    save_graph_to_json(None, n_noise, i+1, folder=path)

def test_synthetic_graph_generation(starting_nodes, functions, max_n_noise=20, iterations=20, dataset_size=1000, n_real_nodes=10, folder='test', nalingam_score_iterations=30, nalingam_graph_iterations=20, non_linear=False):
    """
    Tests various graph discovery methods for graph composition on the real dataset with added noise variables.

    Parameters:
    - starting_nodes (int): The number of initial nodes to include in the subgraph.
    - functions (list): A list of graph discovery method names as strings.
    - max_n_noise (int): The maximum number of noise variables to add.
    - iterations (int): The number of iterations to run for each noise level.
    - dataset_size (int): The number of samples in the synthetic dataset.
    - n_nodes (int): The total number of nodes in the synthetic graph.
    - folder (str): The folder to save the results in.

    Saves the discovered and real graphs for each method, noise level and iteration.
    """

    for n_noise in range(max_n_noise + 1):
        print(f"Generating dataset with {n_noise} noise variables")
        
        dataset = SyntheticGraphGenerator(
            sample_size=dataset_size,
            node_size=n_real_nodes + n_noise,
            min_edges=2,
            max_edges=5,
            n_noise_nodes=n_noise,
            non_linear=non_linear
            )

        for i in range(iterations):
            dataset.generate_samples()

            df = dataset.get_dataframe()
            nodes = dataset.get_nodes()
            real_graph = dataset.get_graph()
            
            
            path = f'results_synthetic/{folder}/real_graph'
            if not os.path.exists(path):
                os.makedirs(path)
            save_graph_to_json(real_graph, n_noise, i+1, folder=path)

            if (i+1) % 10 == 0:
                print(f"Iteration {i+1}/{iterations}")

            initial_variables = nodes[:starting_nodes]
            initial_graph = get_initial_subgraph(real_graph, initial_variables)

            for discovery_method in functions:
                print(f"Running discovery method: {discovery_method}")
                if discovery_method == 'NALiNGAMAlgorithm':
                    env = GraphEnvNALiNGAM(df, initial_graph)
                    found_state, _ = env.get_best_state_fast(nalingam_score_iterations)

                    graph = env.get_graph(found_state, iterations=nalingam_graph_iterations)
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
                path = f'results_synthetic/{folder}/{discovery_method}'
                # If folder does not exist, create it
                if not os.path.exists(path):
                    os.makedirs(path)
                save_graph_to_json(graph, n_noise, i+1, folder=path)

def evaluate_graphs(graphs_dict, real_graphs, is_real_data):
    """
    Evaluates the performance of various graph discovery methods by comparing their discovered graphs to the real graphs.

    Parameters:
    - graphs_dict (dict): A dictionary where keys are method names and values are dictionaries with noise levels as keys and lists of discovered graphs (networkx.DiGraph) as values.
    - real_graphs (dict): A dictionary where keys are noise levels/iterations (or 'real' for real data) and values are the corresponding real graphs (networkx.DiGraph).
    - is_real_data (bool): A flag indicating whether the data is real or synthetic.
    Returns:
    - dict: A dictionary containing evaluation metrics (AUC, SID, SHD, Edges Accuracy, Invented Edges Accuracy, Invented Wrong Edges Accuracy) for each method and noise level.
    """

    results = {}

    if is_real_data:
        real_graph = real_graphs['real']

    for method in graphs_dict.keys():
        results[method] = {}

    for method, graph_info in graphs_dict.items():
        method_results = {'AUC': [], 'SID': [], 'SHD': [], 'Edges Accuracy': [],
                         'Invented Edges Accuracy': [], 'Invented Wrong Edges Accuracy': []}
        for noise_level in graph_info.keys():
            if method == 'GrangerAlgorithm':
                continue
            print(f"Evaluating method: {method}, Noise level: {noise_level}")

            graphs = graph_info[noise_level]

            if len(graphs) > 0:
                temp_auc = 0
                temp_sid = 0
                temp_shd = 0
                edges_accuracy = 0
                invented_edges_accuracy = 0
                invented_wrong_edges_accuracy = 0
                for i, graph in enumerate(graphs):
                    if not is_real_data:
                        real_graph = real_graphs[noise_level][i]

                    if graph is None:
                        edges_accuracy += 0
                        invented_edges_accuracy += 1
                        invented_wrong_edges_accuracy += 1
                        temp_auc += 0
                        temp_sid += len(real_graph.nodes()) * (len(real_graph.nodes()) - 1) / 2
                        temp_shd += len(real_graph.edges())

                    else:
                        nodes_test = list(set(list(real_graph.nodes()) + list(graph.nodes())))

                        graph_test = nx.DiGraph()
                        graph_test.add_nodes_from(nodes_test)
                        graph_test.add_edges_from(graph.edges())

                        real_graph_test = nx.DiGraph()
                        real_graph_test.add_nodes_from(nodes_test)
                        real_graph_test.add_edges_from(real_graph.edges())

                        temp_edges_accuracy, temp_invented_edges_accuracy, temp_invented_wrong_edges_accuracy = check_graph(graph_test, real_graph_test)
                        edges_accuracy += temp_edges_accuracy
                        invented_edges_accuracy += temp_invented_edges_accuracy
                        invented_wrong_edges_accuracy += temp_invented_wrong_edges_accuracy

                        temp_auc += precision_recall(real_graph_test, graph_test)[0]
                        temp_shd += SHD(real_graph_test, graph_test)

                        # Add nodes from real_graph to the graph if they are missing
                        missing_nodes = set(real_graph.nodes()) - set(graph.nodes())
                        for node in missing_nodes:
                            graph.add_node(node)
                        # predicted_scores = [metric(real_graph_test, graph_test) for metric in (precision_recall, SID, SHD)]
                        # temp_auc += predicted_scores[0][0]
                        # temp_sid += predicted_scores[1]
                        # temp_shd += predicted_scores[2]

                        temp_sid += SID(real_graph_test, graph_test)

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

def test_real_ablation(initial_graph, parameters, max_n_noise=20, iterations=20, folder='test', dataset_class=RealDataSachs, nalingam_score_iterations=30, nalingam_graph_iterations=20):
    """
    Tests various ablation settings for the NALiNGAM graph discovery method on the real dataset with added noise variables.

    Parameters:
    - initial_graph (networkx.DiGraph): The initial subgraph to start the discovery from.
    - parameters (list): A list of dictionaries containing parameters for each graph discovery method.
    - max_n_noise (int): The maximum number of noise variables to add.
    - iterations (int): The number of iterations to run for each noise level.
    - folder (str): The folder to save the results in.

    Saves the real graph only once when n_noise is 0 and iteration is 0.
    Saves the discovered graphs for each method, noise level and iteration.
    """

    for n_noise in range(0, max_n_noise + 1):
        print(f"Generating dataset with {n_noise} noise variables")
        
        for i in range(iterations):
            dataset = dataset_class(n_noise=n_noise)
            df = dataset.get_dataframe()
            
            if n_noise == 0 and i == 0:
                # Save the real graph only once
                path = f'results_real/{folder}/real_graph'
                if not os.path.exists(path):
                    os.makedirs(path)
                save_graph_to_json(dataset.get_graph(), n_noise, i+1, folder=path)

            if (i+1) % 10 == 0:
                print(f"Iteration {i+1}/{iterations}")
            for param in parameters:
                ablation_method = param['name']
                try:
                    print(f"Running discovery method: {ablation_method}")

                    if 'p_reg' not in param:
                        param['p_reg'] = 1
                    if 'p_perm' not in param:
                        param['p_perm'] = 1
                    if 'p_boots' not in param:
                        param['p_boots'] = 1
                    if 'p_miss' not in param:
                        param['p_miss'] = 1
                    if 'p_miss_penalty' not in param:
                        param['p_miss_penalty'] = 10
                    if 'permutation_iterations' not in param:
                        param['permutation_iterations'] = 10
                    if 'bootstrap_iterations' not in param:
                        param['bootstrap_iterations'] = 10

                    env = GraphEnvNALiNGAM(df, initial_graph, p_reg=param['p_reg'], p_perm=param['p_perm'], p_boots=param['p_boots'], p_miss=param['p_miss'], p_miss_penalty=param['p_miss_penalty'], permutation_iterations=param['permutation_iterations'], bootstrap_iterations=param['bootstrap_iterations'])
                    found_state, _ = env.get_best_state_fast(nalingam_score_iterations)

                    graph = env.get_graph(found_state, iterations=nalingam_graph_iterations)


                    path = f'results_real/{folder}/{ablation_method}'
                    # If folder does not exist, create it
                    if not os.path.exists(path):
                        os.makedirs(path)
                    save_graph_to_json(graph, n_noise, i+1, folder=path)
                except Exception as e:
                    print(f"Error running discovery method {ablation_method} with n_noise={n_noise} and iteration={i+1}: {e}")
                    path = f'results_real/{folder}/{ablation_method}'
                    save_graph_to_json(None, n_noise, i+1, folder=path)