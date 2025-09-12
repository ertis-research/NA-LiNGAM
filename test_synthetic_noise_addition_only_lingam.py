####################################
############# LIBRARIES ############
####################################

from utils.graph_functions import test_synthetic_graph_generation


##############################################
##############################################

functions_list = [
   'NALiNGAMAlgorithm',
   'LiNGAMAlgorithm'
]

test_synthetic_graph_generation(3, functions_list, iterations=20, folder='results_lingam_20_noise_20_iter')