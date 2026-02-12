####################################
############# LIBRARIES ############
####################################

import networkx as nx
from utils.generate_real_data_graph import AdaByronData
from utils.graph_functions import test_real_graph_generation


##############################################
##############################################

initial_variables = ['Consumption_Wh', 'AC_A_Energy_Wh', 'AC_B_Energy_Wh']

initial_graph = nx.DiGraph()
initial_graph.add_nodes_from(initial_variables)
initial_graph.add_edges_from([
    ('AC_A_Energy_Wh', 'Consumption_Wh'),
    ('AC_B_Energy_Wh', 'Consumption_Wh')
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
   'GrangerAlgorithm',
   # 'NOTEARSAlgorithm',
   'CAMAlgorithm',
   # 'CCDrAlgorithm',
   'GIESAlgorithm'
]

test_real_graph_generation(initial_graph, functions_list, iterations=20, folder='results_ada_all_20_noise_20_iter', dataset_class=AdaByronData, nalingam_score_iterations=10, nalingam_graph_iterations=1)