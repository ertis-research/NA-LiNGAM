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

class NALiNGAMAlgorithm():
    """
    Validation algorithm for LiNGAM graph.
    params:
        data: DataFrame with the data.
        variables: List with the variables to consider.
    """
    def __init__(self, data, variables):
        self.data = data
        self.variables = variables

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
        Validate the edges of the LiNGAM graph using conditional regression.
        return:
            results: Dictionary with the p-values of each edge.
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
        Validate the edges of the LiNGAM graph comparing the observed relationship with a null distribution based on random permutations.
        params:
            num_permutations: Number of permutations to perform.
        return:
            results: Dictionary with the p-values of each edge.
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

    def validate_edges_bootstrap(self, num_samples=10):
        """
        Validate the edges of the LiNGAM graph by seeing how many times they appear in different subsets (bootstrapping).
        params:
            num_samples: Number of samples to consider.
        return:
            results: Dictionary with the probabilities of each edge.
        """
        data = self.data[self.variables]
        num_vars = data.shape[1]
        adjacency_counts = np.zeros((num_vars, num_vars))  # Matriz para contar apariciones de aristas

        for _ in range(num_samples):
            sample_data = data.sample(frac=1, replace=True)  # Bootstrap con reemplazo
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
        Verifies if a node has relationships with other nodes within the graph.
        params:
            node: Node to verify.
        return:
            True if it has relationships, False otherwise.
        """
        return len(list(self.graph.predecessors(node))) > 0 or len(list(self.graph.successors(node))) > 0

    def get_nodes(self):
        """"
        Get the graph nodes.
        return:
            nodes: List of the graph nodes.
        """
        return list(self.graph.nodes())
        
    def get_edges(self):
        """"
        Get the graph edges."
        return:
            edges: List of the graph edges.
        """
        return list(self.graph.edges())

    def get_graph(self):
        """"
        Get the LiNGAM graph.
        return:
            graph: LiNGAM graph.
        """
        return self.graph
    
    def get_sorted_edges(self):
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
        """"
        Get the validation score of the LiNGAM graph.
        params:
            num_iter: Number of iterations of the algorithm.
        return:
            total_score: Validation score of the LiNGAM graph.
        """
        if num_iter is None:
            num_iter = 1

        total_score = 0
        for _ in range(num_iter):
            results_regression = self.validate_edges_regression()
            results_permutation = self.validate_edges_permutation(num_permutations=10)
            results_bootstrap = self.validate_edges_bootstrap(num_samples=10)
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
                score = ((1 - reg_score) + (1 - perm_score)) * (boots_score / 2)
                sample_score += score

            total_new_edges = 0
            prob_score = 0
            for (x, y) in results_bootstrap.keys():
                if (x, y) not in results_regression.keys():
                    if not self.has_relations(x) or not self.has_relations(y):
                        if show: print(f"Edge: {x} -> {y}: Boots: {results_bootstrap[(x, y)]} BAD EDGE")
                        prob_score += 10
                    else:
                        if show: print(f"Edge: {x} -> {y}: Boots: {results_bootstrap[(x, y)]}")
                        prob_score += results_bootstrap[(x, y)]
                    total_new_edges += 1
            if total_new_edges > 0:
                prob_score = prob_score / total_new_edges
            sample_score -= prob_score

            total_score += sample_score

        total_score = total_score / num_iter
        return total_score