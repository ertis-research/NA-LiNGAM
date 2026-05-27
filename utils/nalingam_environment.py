##############################################
############## IMPORT LIBRARIES ##############
##############################################

from .nalingam_score import NALiNGAMScoreAlgorithm

import numpy as np
import pandas as pd

import networkx as nx

from copy import deepcopy

import concurrent.futures

##############################################
############## CLASS DEFINITION ##############
##############################################

class GraphEnvNALiNGAM:
    def __init__(self, data, initial_graph, p_reg=1, p_perm=1, p_boots=1, p_miss=1, p_miss_penalty=10, permutation_iterations=10, bootstrap_iterations=10):
        """
        Environment for the NALiNGAM algorithm.

        Parameters:
        - data (pd.DataFrame): The dataset to be used.
        - initial_graph (nx.DiGraph): The initial graph structure.
        """
        self.data = data
        self.initial_graph = initial_graph
        self.init_vars = list(initial_graph.nodes())
        
        self.new_features = []

        for node in self.data.columns.tolist():
            if node not in list(self.init_vars):
                self.new_features.append(node)

        self.initial_state = np.zeros(len(self.new_features))

        self.p_reg = p_reg
        self.p_perm = p_perm
        self.p_boots = p_boots
        self.p_miss = p_miss
        self.p_miss_penalty = p_miss_penalty
        self.permutation_iterations = permutation_iterations
        self.bootstrap_iterations = bootstrap_iterations
##############################################
############ GETTERS AND SETTERS #############
##############################################

    def get_initial_state(self):
        """
        Returns the initial state of the environment.
        """
        return self.initial_state
    
    def get_possible_states(self):
        """
        Returns all possible states of the environment."""
        states = []
        for i in range(2**len(self.initial_state)):
            states.append([int(x) for x in list(bin(i)[2:].zfill(len(self.initial_state)))])
        return states
    
    def get_graph(self, state, iterations=10):
        """
        Returns the graph corresponding to the given state.

        Parameters:
        - state (list): The state to be evaluated.
        - iterations (int): Number of iterations to run the NALiNGAM algorithm.

        Returns:
        - graph (nx.DiGraph): The resulting graph.
        """
        selected_features = [feature for feature, mask in zip(self.new_features, state) if mask == 1]

        selected_lingam = None
        best_score = -np.inf

        for _ in range(iterations):
            self.lingam = NALiNGAMScoreAlgorithm(self.data, self.init_vars + selected_features, p_reg=self.p_reg, p_perm=self.p_perm, p_boots=self.p_boots, p_miss=self.p_miss, p_miss_penalty=self.p_miss_penalty, permutation_iterations=self.permutation_iterations, bootstrap_iterations=self.bootstrap_iterations)
            score = self.lingam.get_score(num_iter=1)

            if score > best_score:
                best_score = score
                selected_lingam = self.lingam
        self.lingam = selected_lingam

        graph = nx.DiGraph()
        graph.add_nodes_from(list(self.lingam.graph.nodes()))
        graph.add_edges_from(list(self.initial_graph.edges()))

        # Add edges from the LiNGAM graph only if they do not break the acyclic property
        for edge in self.lingam.get_sorted_edges():
            temp_graph = deepcopy(graph)
            temp_graph.add_edge(edge[0], edge[1])
            if nx.is_directed_acyclic_graph(temp_graph):
                graph.add_edge(edge[0], edge[1])

        return graph
    
##############################################
############## SCORE FUNCTIONS ###############
##############################################

    def get_score(self, state, num_iter=None, show=False):
        """
        Returns the score of the given state.

        Parameters:
        - state (list): The state to be evaluated.
        - num_iter (int): Number of iterations to run the NALiNGAM algorithm
        - show (bool): Whether to print the score.

        Returns:
        - score (float): The score of the state.
        """
        selected_features = [feature for feature, mask in zip(self.new_features, state) if mask == 1]
        
        lingam = NALiNGAMScoreAlgorithm(self.data, self.init_vars + selected_features, p_reg=self.p_reg, p_perm=self.p_perm, p_boots=self.p_boots, p_miss=self.p_miss, p_miss_penalty=self.p_miss_penalty, permutation_iterations=self.permutation_iterations, bootstrap_iterations=self.bootstrap_iterations)

        score = lingam.get_score(num_iter=num_iter, show=show)

        return score
    
    def get_best_state_fast(self, num_iter = None, show=False):
        """
        Returns the best state found using a fast search strategy.

        Parameters:
        - num_iter (int): Number of iterations to run the NALiNGAM algorithm
        - show (bool): Whether to print the score.

        Returns:
        - best_state (list): The best state found.
        - best_score (float): The score of the best state.
        """
        
        initial_states = []
        initial_score = self.get_score(self.initial_state, num_iter=num_iter)

        # Get states with one permutation of the initial state
        for i in range(len(self.initial_state)):
            new_state = deepcopy(self.initial_state)
            new_state[i] = 1 - new_state[i]
            initial_states.append(new_state)

        # Get scores for the initial states
        with concurrent.futures.ProcessPoolExecutor() as executor:
            initial_scores = executor.map(self.get_score, initial_states, [num_iter]*len(initial_states))

        # Get indexes of better scores
        indexes = [i for i, score in enumerate(initial_scores) if score > initial_score]

        if len(indexes) == 0:
            return self.initial_state, initial_score
        
        state = deepcopy(self.initial_state)
        for i in indexes:
            state[i] = 1

        temp_graph = self.get_graph(state)

        # Delete nodes without relations
        # If a node has no relations, it is not part of the best state
        for node in temp_graph.nodes():
            if temp_graph.degree(node) == 0:
                state[self.new_features.index(node)] = 0

        return state, self.get_score(state, show=show)
    
    def get_best_state(self, num_iter=None):
        """
        Returns the best state found using a complete search strategy.

        Parameters:
        - num_iter (int): Number of iterations to run the NALiNGAM algorithm

        Returns:
        - best_state (list): The best state found.
        - best_score (float): The score of the best state.
        """
        states = self.get_possible_states()
        best_state = None
        best_score = - np.inf

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(self.get_score, states, [num_iter]*len(states))

        best_state, _ = max(zip(states, results), key=lambda x: x[1])

        temp_graph = self.get_graph(best_state)

        # Delete nodes without relations
        # If a node has no relations, it is not part of the best state
        for node in temp_graph.nodes():
            if temp_graph.degree(node) == 0:
                best_state[self.new_features.index(node)] = 0

        return best_state, self.get_score(best_state)