####################################
############# LIBRARIES ############
####################################

from utils.graph_functions import test_synthetic_graph_generation


##############################################
##############################################

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
   'NOTEARSAlgorithm',
   'CAMAlgorithm',
   #'CCDrAlgorithm',
   'GIESAlgorithm'
]

test_synthetic_graph_generation(3, functions_list, iterations=20, folder='results_nonlinear_all_20_noise_20_iter', non_linear=True)