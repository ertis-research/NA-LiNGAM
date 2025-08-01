####################################
############# LIBRARIES ############
####################################

from utils.environment import GraphEnvLiNGAM

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
from utils.na_lingam import NALiNGAMAlgorithm

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
                save_graph_to_json(initial_graph, n_noise, i+1, folder=path)

            if (i+1) % 10 == 0:
                print(f"Iteration {i+1}/{iterations}")
            for discovery_method in functions:
                print(f"Running discovery method: {discovery_method}")
                if discovery_method == 'NALiNGAMAlgorithm':
                    env = GraphEnvLiNGAM(df, initial_graph)
                    found_state, _ = env.get_best_state_fast(30)

                    graph = env.get_graph(found_state, iterations=20)
                elif discovery_method == 'LiNGAMAlgorithm':
                    env = GraphEnvLiNGAM(df, initial_graph)

                    graph = env.get_graph(np.ones(len(df.columns) - len(list(initial_graph.nodes()))), iterations=1)
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

            if i == 0:
                path = f'results_synthetic/{folder}/real_graph'
                if not os.path.exists(path):
                    os.makedirs(path)
                save_graph_to_json(initial_graph, n_noise, i+1, folder=path)

            df = dataset.get_dataframe()

            for discovery_method in functions:
                print(f"Running discovery method: {discovery_method}")
                if discovery_method == 'NALiNGAMAlgorithm':
                    env = GraphEnvLiNGAM(df, initial_graph)
                    found_state, _ = env.get_best_state_fast(30)

                    graph = env.get_graph(found_state, iterations=20)
                elif discovery_method == 'LiNGAMAlgorithm':
                    env = GraphEnvLiNGAM(df, initial_graph)

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