########################################################
################## IMPORT LIBRARIES ####################
########################################################

import numpy as np
import networkx as nx
from causallearn.search.FCMBased import lingam
import json
import math

from cdt.metrics import precision_recall, SID, SHD

from utils.generate_synthetic_graph import SyntheticGraphGenerator
from utils.graph_functions import get_initial_subgraph
from utils.nalingam_environment import GraphEnvNALiNGAM
from utils.nalingam_score import NALiNGAMScoreAlgorithm

def bootstrap_validation(df, nodes, num_samples=100):

    nalingam_score = NALiNGAMScoreAlgorithm(df, nodes)
    results = nalingam_score.validate_edges_bootstrap(num_samples=num_samples, with_replacement=True)

    final_edges = {edge: prob for edge, prob in results.items() if prob >= 0.85}
    
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(final_edges.keys())

    # Remove nodes without relations
    node_list = list(graph.nodes())

    for node in node_list:
        if len(list(graph.predecessors(node))) == 0 and len(list(graph.successors(node))) == 0:
            graph.remove_node(node)
    
    return graph

def stability_selection(df, nodes, num_samples=100):
    data = df[nodes]
    num_vars = data.shape[1]
    adjacency_counts = np.zeros((num_vars, num_vars))
    q_sum = 0

    for _ in range(num_samples):
        sample_data = data.sample(frac=0.5, replace=False)  
        
        model = lingam.ICALiNGAM().fit(sample_data)
        discovered_edges = (model.adjacency_matrix_ != 0)
        
        adjacency_counts += discovered_edges
        q_sum += np.sum(discovered_edges)

    adjacency_probs = adjacency_counts / num_samples

    q = q_sum / num_samples
    p_total = num_vars * (num_vars - 1)

    false_positives_threshold = math.sqrt(num_vars)

    pi_th = 0.5 + (q**2) / (2 * false_positives_threshold * p_total)

    pi_th = np.clip(pi_th, 0.51, 0.99)

    # print(f"Threshold: {pi_th:.2f}")

    results = {}
    for i, j in zip(*np.where(adjacency_probs != 0)):
        results[(data.columns[j], data.columns[i])] = adjacency_probs[i, j]

    final_edges = {edge: prob for edge, prob in results.items() if prob >= pi_th}

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(final_edges.keys())

    # Remove nodes without relations
    node_list = list(graph.nodes())
    for node in node_list:
        if len(list(graph.predecessors(node))) == 0 and len(list(graph.successors(node))) == 0:
            graph.remove_node(node)
    
    return graph

def add_missing_nodes(graph, real_graph):
    missing_nodes = set(real_graph.nodes()) - set(graph.nodes())
    for node in missing_nodes:
        graph.add_node(node)

    return graph

def test_algorithm(syn_graph, n_tests=10, starting_nodes=3):
    
    nalingam_auc_mean = 0
    bootstrap_auc_mean = 0
    stability_auc_mean = 0

    nalingam_shd_mean = 0
    bootstrap_shd_mean = 0
    stability_shd_mean = 0

    nalingam_sid_mean = 0
    bootstrap_sid_mean = 0
    stability_sid_mean = 0

    for _ in range(n_tests):
        syn_graph.generate_samples()

        real_nodes, _ = syn_graph.get_real_noise_nodes()
        initial_variables = real_nodes[:starting_nodes]

        dataset = syn_graph.get_dataframe()

        real_variables, _ = syn_graph.get_real_noise_nodes()
        real_graph = syn_graph.get_graph()
        initial_variables = real_variables[:starting_nodes]
        initial_graph = get_initial_subgraph(syn_graph.get_graph(), initial_variables)

        env = GraphEnvNALiNGAM(dataset, initial_graph)
        found_state, _ = env.get_best_state_fast(30)
        nalingam_graph = env.get_graph(found_state, iterations=20)

        bootstrap_graph = bootstrap_validation(dataset, syn_graph.get_nodes(), num_samples=100)
        stability_graph = stability_selection(dataset, syn_graph.get_nodes(), num_samples=100)

        nalingam_graph = add_missing_nodes(nalingam_graph, real_graph)
        bootstrap_graph = add_missing_nodes(bootstrap_graph, real_graph)
        stability_graph = add_missing_nodes(stability_graph, real_graph)

        nalingam_auc_mean += precision_recall(real_graph, nalingam_graph)[0]
        bootstrap_auc_mean += precision_recall(real_graph, bootstrap_graph)[0]
        stability_auc_mean += precision_recall(real_graph, stability_graph)[0]

        nalingam_shd_mean += SHD(real_graph, nalingam_graph)
        bootstrap_shd_mean += SHD(real_graph, bootstrap_graph)
        stability_shd_mean += SHD(real_graph, stability_graph)

        nalingam_sid_mean += SID(real_graph, nalingam_graph)
        bootstrap_sid_mean += SID(real_graph, bootstrap_graph)
        stability_sid_mean += SID(real_graph, stability_graph)

    nalingam_results = {
        'auc': nalingam_auc_mean / n_tests,
        'shd': nalingam_shd_mean / n_tests,
        'sid': nalingam_sid_mean / n_tests
    }

    bootstrap_results = {
        'auc': bootstrap_auc_mean / n_tests,
        'shd': bootstrap_shd_mean / n_tests,
        'sid': bootstrap_sid_mean / n_tests
    }

    stability_results = {
        'auc': stability_auc_mean / n_tests,
        'shd': stability_shd_mean / n_tests,
        'sid': stability_sid_mean / n_tests
    }

    return nalingam_results, bootstrap_results, stability_results

if __name__ == "__main__":
    dataset_size = 1000
    n_tests = 50
    n_real_nodes = 10
    max_n_noise = 20
    starting_nodes = 3

    nalingam_auc_history = []
    bootstrap_auc_history = []
    stability_auc_history = []

    nalingam_shd_history = []
    bootstrap_shd_history = []
    stability_shd_history = []

    nalingam_sid_history = []
    bootstrap_sid_history = []
    stability_sid_history = []

    for n_noise in range(max_n_noise + 1):
        print(f"Generating dataset with {n_noise} noise variables")
        syn_graph = SyntheticGraphGenerator(
            sample_size=dataset_size,
            node_size=n_real_nodes + n_noise,
            min_edges=2,
            max_edges=5,
            n_noise_nodes=n_noise
            )

        nalingam_results, bootstrap_results, stability_results = test_algorithm(syn_graph, n_tests, starting_nodes)

        nalingam_auc_history.append(nalingam_results['auc'])
        bootstrap_auc_history.append(bootstrap_results['auc'])
        stability_auc_history.append(stability_results['auc'])

        nalingam_shd_history.append(nalingam_results['shd'])
        bootstrap_shd_history.append(bootstrap_results['shd'])
        stability_shd_history.append(stability_results['shd'])

        nalingam_sid_history.append(nalingam_results['sid'])
        bootstrap_sid_history.append(bootstrap_results['sid'])
        stability_sid_history.append(stability_results['sid'])

        results_auc = {
            'NA-LiNGAM': nalingam_auc_history, 
            'Bootstrap': bootstrap_auc_history, 
            'Stability': stability_auc_history
        }

        results_shd = {
            'NA-LiNGAM': nalingam_shd_history, 
            'Bootstrap': bootstrap_shd_history, 
            'Stability': stability_shd_history
        }

        results_sid = {
            'NA-LiNGAM': nalingam_sid_history, 
            'Bootstrap': bootstrap_sid_history, 
            'Stability': stability_sid_history
        }

        with open(f'graph_metrics/bootstrap_validation_auc_50_20.json', 'w') as f:
            json.dump(results_auc, f)

        with open(f'graph_metrics/bootstrap_validation_shd_50_20.json', 'w') as f:
            json.dump(results_shd, f)

        with open(f'graph_metrics/bootstrap_validation_sid_50_20.json', 'w') as f:
            json.dump(results_sid, f)

        print('Finished testing')