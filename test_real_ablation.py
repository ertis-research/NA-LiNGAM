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
   {'name': 'FULL', 'p_reg' : 1, 'p_perm' : 1, 'p_boots' : 1, 'p_miss' : 1, 'p_miss_penalty' : 10},
    {'name': 'NO_REG', 'p_reg' : 0, 'p_perm' : 1, 'p_boots' : 1, 'p_miss' : 1, 'p_miss_penalty' : 10},
    {'name': 'NO_PERM', 'p_reg' : 1, 'p_perm' : 0, 'p_boots' : 1, 'p_miss' : 1, 'p_miss_penalty' : 10},
    {'name': 'NO_BOOTS', 'p_reg' : 1, 'p_perm' : 1, 'p_boots' : 0, 'p_miss' : 1, 'p_miss_penalty' : 10},
    {'name': 'NO_MISS', 'p_reg' : 1, 'p_perm' : 1, 'p_boots' : 1, 'p_miss' : 0, 'p_miss_penalty' : 10},
    {'name': 'NO_MISS_PENALTY', 'p_reg' : 1, 'p_perm' : 1, 'p_boots' : 1, 'p_miss' : 1, 'p_miss_penalty' : 0}
]

test_real_ablation(initial_graph, parameter_list, iterations=20, folder='results_ada_ablation_20_noise_20_iter', dataset_class=AdaByronData, nalingam_score_iterations=10, nalingam_graph_iterations=1)