import numpy as np
from utils.nalingam_environment import GraphEnvNALiNGAM
from utils.generate_synthetic_graph import SyntheticGraphGenerator
from utils.graph_functions import get_initial_subgraph
from utils.nalingam_score import NALiNGAMScoreAlgorithm

from causallearn.search.FCMBased import lingam
import networkx as nx

######################################################################
######################################################################

starting_nodes = 3

dataset = SyntheticGraphGenerator(
    sample_size=1000,
    node_size=20,
    min_edges=2,
    max_edges=5,
    n_noise_nodes=10
)

dataset.generate_samples()

df = dataset.get_dataframe()
nodes = dataset.get_nodes()
print("Nodes:", nodes)
real_graph = dataset.get_graph()

initial_variables = nodes[:starting_nodes]
initial_graph = get_initial_subgraph(real_graph, initial_variables)

env = GraphEnvNALiNGAM(df, initial_graph)

######################################################################
######################################################################

# NA-LiNGAM

print("Running NA-LiNGAM...")

found_state, _ = env.get_best_state_fast(30)
nalingam_graph = env.get_graph(found_state, iterations=20)

######################################################################
######################################################################

# Bootstrap validation

print("Running Bootstrap Validation...")

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

bootstrap_graph = bootstrap_validation(df, nodes, num_samples=100)

######################################################################
######################################################################

# Stability Selection

print("Running Stability Selection...")

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
        q_sum += np.sum(discovered_edges)  # Contamos las aristas encontradas en esta iteración

    adjacency_probs = adjacency_counts / num_samples

    q = q_sum / num_samples  # Promedio de aristas descubiertas por subsample
    p_total = num_vars * (num_vars - 1)  # Total de conexiones posibles en el grafo

    # Tú decides cuántas aristas falsas (falsos positivos) estás dispuesto a tolerar como máximo
    falsos_positivos_tolerados = 1.0  

    # La fórmula teórica calcula el umbral mínimo (pi_th) necesario para garantizar esa tolerancia
    # Fórmula original: E(V) <= (q^2) / ((2 * pi_th - 1) * p_total)
    pi_th = 0.5 + (q**2) / (2 * falsos_positivos_tolerados * p_total)

    # Nos aseguramos de que el umbral tenga sentido estadístico (entre 0.5 y 1.0)
    pi_th = np.clip(pi_th, 0.51, 0.99)

    print(f"El umbral calculado matemáticamente es: {pi_th:.2f}")

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

stability_graph = stability_selection(df, nodes, num_samples=100)

######################################################################
######################################################################

from cdt.metrics import precision_recall, SID, SHD

print("NA-LiNGAM:")
print("Precision and Recall:", precision_recall(real_graph, nalingam_graph)[0])
print("SID:", SID(real_graph, nalingam_graph))
print("SHD:", SHD(real_graph, nalingam_graph))

print("\nBootstrap Validation:")
print("Precision and Recall:", precision_recall(real_graph, bootstrap_graph)[0])
print("SID:", SID(real_graph, bootstrap_graph))
print("SHD:", SHD(real_graph, bootstrap_graph))

print("\nStability Selection:")
print("Precision and Recall:", precision_recall(real_graph, stability_graph)[0])
print("SID:", SID(real_graph, stability_graph))
print("SHD:", SHD(real_graph, stability_graph))