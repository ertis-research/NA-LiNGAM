####################################
############# LIBRARIES ############
####################################

import networkx as nx
from utils.graph_functions import test_synthetic_graph_generation


##############################################
##############################################

initial_variables = ['PKC', 'PKA', 'Akt']

initial_graph = nx.DiGraph()
initial_graph.add_nodes_from(initial_variables)
initial_graph.add_edges_from([
    ('PKC', 'PKA'),
    ('PKA', 'Akt')
])

functions_list = [
   'NALiNGAMAlgorithm',
   'PCAlgorithm',
   'FCIAlgorithm',
   'GESAlgorithm',
   'BOSSAlgorithm',
   'ExactSearchAlgorithm',
   'LiNGAMAlgorithm',
   'GRaSPAlgorithm',
   'GrangerAlgorithm'
]

test_synthetic_graph_generation(3, functions_list, iterations=20, folder='results_all_20_noise_20_iter')