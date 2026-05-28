####################################
############# LIBRARIES ############
####################################

import networkx as nx
from utils.generate_real_data_graph import AdaByronData
from utils.graph_functions import test_real_ablation


##############################################
##############################################

initial_variables = ['Consumption_Wh', 'AC_A_Energy_Wh', 'AC_B_Energy_Wh']

initial_graph = nx.DiGraph()
initial_graph.add_nodes_from(initial_variables)
initial_graph.add_edges_from([
    ('AC_A_Energy_Wh', 'Consumption_Wh'),
    ('AC_B_Energy_Wh', 'Consumption_Wh')
])

parameter_list = [
    {'name': 'PERMUTATION=5', 'permutation_iterations': 5},
    {'name': 'PERMUTATION=10', 'permutation_iterations': 10},
    {'name': 'PERMUTATION=20', 'permutation_iterations': 20},
    {'name': 'PERMUTATION=50', 'permutation_iterations': 50}
]

test_real_ablation(initial_graph, parameter_list, iterations=5, folder='results_ada_ablation_permutation_20_noise_20_iter', dataset_class=AdaByronData, nalingam_score_iterations=10, nalingam_graph_iterations=1)