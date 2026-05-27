##############################################
############## IMPORT LIBRARIES ##############
##############################################

import dowhy
from dowhy import CausalModel

import numpy as np
import pandas as pd
import graphviz
import networkx as nx
import pydotplus

import matplotlib.pyplot as plt

from causallearn.search.FCMBased import lingam
from causallearn.search.FCMBased.lingam.utils import make_dot

import statsmodels.api as sm
from scipy.stats import pearsonr

##############################################
############## CLASS DEFINITION ##############
##############################################

class NALiNGAMScoreAlgorithm():
    
    def __init__(self, data, variables, p_reg=1, p_perm=1, p_boots=1, p_miss=1, p_miss_penalty=10, permutation_iterations=10, bootstrap_iterations=10):
        """
        Initializes the NALiNGAMScoreAlgorithm with the given data and variables.

        Parameters:
        - data (pd.DataFrame): The dataset to be used.
        - variables (list): List of variable names to be considered in the model.
        """
        self.data = data
        self.variables = variables

        self.p_reg = p_reg
        self.p_perm = p_perm
        self.p_boots = p_boots
        self.p_miss = p_miss
        self.p_miss_penalty = p_miss_penalty

        self.permutation_iterations = permutation_iterations
        self.bootstrap_iterations = bootstrap_iterations

        self.model = lingam.ICALiNGAM()
        self.model.fit(self.data[self.variables])

        dot = make_dot(self.model.adjacency_matrix_, labels=variables)
        dotplus = pydotplus.graph_from_dot_data(dot.source)
        init_graph = nx.nx_pydot.from_pydot(dotplus)

        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(variables)
        self.graph.add_edges_from(init_graph.edges())

    ##############################################
    ############ VALIDATION FUNCTIONS ############
    ##############################################

    def validate_edges_regression(self):
        """
        Validates the edges of the LiNGAM graph using regression analysis.

        Returns:
        - results (dict): A dictionary with edges as keys and their corresponding p-values as values
        """
        
        data = self.data[self.variables]
        results = {}
        for i, j in zip(*np.where(self.model.adjacency_matrix_ != 0)):  # Solo donde hay aristas
            X = data.iloc[:, i]
            Y = data.iloc[:, j]
            other_vars = [k for k in range(data.shape[1]) if k != i and k != j]
            
            # Construir el modelo de regresión
            X_other = data.iloc[:, other_vars]
            X_other = sm.add_constant(X_other)  # Agregar intercepto
            model_reg = sm.OLS(Y, sm.add_constant(pd.concat([X, X_other], axis=1)), missing='drop').fit()
            
            # Guardar resultados
            p_value = model_reg.pvalues[1]  # p-value del coeficiente de X
            results[(data.columns[j], data.columns[i])] = (p_value)

        return results

    def validate_edges_permutation(self, num_permutations=10):
        """
        Validates the edges of the LiNGAM graph using permutation testing.

        Parameters:
        - num_permutations (int): Number of permutations to perform for the test.

        Returns:
        - results (dict): A dictionary with edges as keys and their corresponding p-values as values
        """

        data = self.data[self.variables]
        results = {}
        for i, j in zip(*np.where(self.model.adjacency_matrix_ != 0)):  # Solo donde hay aristas
            if data.columns[i] not in self.variables or data.columns[j] not in self.variables:
                continue
            X = data.iloc[:, i].values
            Y = data.iloc[:, j].values
            
            # Correlación real
            real_corr, _ = pearsonr(X, Y)
            
            # Distribución nula
            null_distr = []
            for _ in range(num_permutations):
                X_perm = np.random.permutation(X)  # Permutar X
                null_corr, _ = pearsonr(X_perm, Y)
                null_distr.append(null_corr)
            
            # p-valor basado en permutaciones
            p_value = (np.sum(np.abs(null_distr) >= np.abs(real_corr)) + 1) / (num_permutations + 1)
            results[(data.columns[j], data.columns[i])] = (p_value)
        
        return results  

    def validate_edges_bootstrap(self, num_samples=10, with_replacement=True):
        """
        Validates the edges of the LiNGAM graph using bootstrap sampling.

        Parameters:
        - num_samples (int): Number of bootstrap samples to generate.

        Returns:
        - results (dict): A dictionary with edges as keys and their corresponding bootstrap frequencies as values
        """

        data = self.data[self.variables]
        num_vars = data.shape[1]
        adjacency_counts = np.zeros((num_vars, num_vars))  # Matriz para contar apariciones de aristas

        for _ in range(num_samples):
            sample_data = data.sample(frac=1, replace=with_replacement)  # Bootstrap con reemplazo
            model = lingam.ICALiNGAM().fit(sample_data)  # Reentrenar modelo
            adjacency_counts += (model.adjacency_matrix_ != 0)  # Sumar aparición de aristas

        adjacency_probs = adjacency_counts / num_samples  # Frecuencia de cada arista

        results = {}
        for i, j in zip(*np.where(adjacency_probs != 0)):
            results[(data.columns[j], data.columns[i])] = adjacency_probs[i, j]

        return results

    ##############################################
    ############## GRAPH FUNCTIONS ###############
    ##############################################

    def has_relations(self, node):
        """
        Checks if a node has any incoming or outgoing edges.

        Parameters:
        - node (str): The node to be checked.

        Returns:
        - has_relations (bool): True if the node has any relations, False otherwise.
        """
        
        return len(list(self.graph.predecessors(node))) > 0 or len(list(self.graph.successors(node))) > 0

    def get_nodes(self):
        """
        Returns a list of all nodes in the graph.
        """
        
        return list(self.graph.nodes())
        
    def get_edges(self):
        """
        Returns a list of all edges in the graph.
        """
        
        return list(self.graph.edges())

    def get_graph(self):
        """
        Returns the current graph.
        """

        return self.graph
    
    def get_sorted_edges(self):
        """
        Returns a sorted list of edges in the graph based on their weights.
        """

        adjacency_matrix = self.model.adjacency_matrix_.T

        pondered_edges = {}
        for i in range(len(adjacency_matrix)):
            for j in range(len(adjacency_matrix)):
                if adjacency_matrix[i, j] != 0:
                    pondered_edges[(self.variables[i], self.variables[j])] = adjacency_matrix[i, j]
        sorted_edges = sorted(pondered_edges.items(), key=lambda x: x[1], reverse=True)

        return [(edge[0][0], edge[0][1]) for edge in sorted_edges]
    
    ##############################################
    ############## SCORE FUNCTIONS ###############
    ##############################################

    def get_score(self, num_iter=None, show=False):
        """
        Computes the score of the current graph based on edge validation results.

        Parameters:
        - num_iter (int): Number of iterations to run the validation functions.
        - show (bool): Whether to print the score.

        Returns:
        - score (float): The computed score of the graph.
        """
        
        if num_iter is None:
            num_iter = 1

        total_score = 0
        for _ in range(num_iter):
            results_regression = self.validate_edges_regression()
            results_permutation = self.validate_edges_permutation(num_permutations=self.permutation_iterations)
            results_bootstrap = self.validate_edges_bootstrap(num_samples=self.bootstrap_iterations)
            if show: print('Selected edges:')
            sample_score = 0
            for (x, y) in results_regression.keys():
                reg_score = results_regression[(x, y)]
                perm_score = results_permutation[(x, y)]
                if (x, y) not in results_bootstrap.keys():
                    boots_score = 0
                else:
                    boots_score = results_bootstrap[(x, y)]
                
                if show: print(f"Edge: {x} -> {y}: Reg: {1 - reg_score}, Perm: {1 - perm_score}, Boots: {boots_score}")
                # score = ((1 - reg_score) + (1 - perm_score)) * (boots_score / 2) # ORIGINAL
                
                if self.p_reg == 0 and self.p_perm == 0:
                    score = 1
                elif self.p_reg == 0 or self.p_perm == 0:
                    score = ((1 - reg_score)*self.p_reg + (1 - perm_score)*self.p_perm)
                else:
                    score = ((1 - reg_score)*self.p_reg + (1 - perm_score)*self.p_perm) / 2

                if self.p_reg == 0 and self.p_perm == 0 and self.p_boots == 0:
                    score = 0
                if self.p_boots != 0:
                    score *= boots_score*self.p_boots

                sample_score += score

            n_miss_edges = 0
            score_miss = 0
            for (x, y) in results_bootstrap.keys():
                if (x, y) not in results_regression.keys():
                    if not self.has_relations(x) or not self.has_relations(y):
                        if show: print(f"Edge: {x} -> {y}: Boots: {results_bootstrap[(x, y)]} BAD EDGE")
                        score_miss += self.p_miss_penalty
                    else:
                        if show: print(f"Edge: {x} -> {y}: Boots: {results_bootstrap[(x, y)]}")
                        score_miss += results_bootstrap[(x, y)] * self.p_miss
                    n_miss_edges += 1
            if n_miss_edges > 0:
                score_miss = score_miss / n_miss_edges
            sample_score -= score_miss

            total_score += sample_score

        total_score = total_score / num_iter
        return total_score