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

test_synthetic_graph_generation(3, functions_list, iterations=100, folder='results_nonlinear_lingam_20_noise_100_iter', non_linear=True)