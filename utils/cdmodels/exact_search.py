##############################################
############## IMPORT LIBRARIES ##############
##############################################

from causallearn.search.ScoreBased.ExactSearch import bic_exact_search
import networkx as nx
import numpy as np

##############################################
############## CLASS DEFINITION ##############
##############################################

class ExactSearchAlgorithm():
    def __init__(self, data, variables):
        self.data = np.array(data[variables])
        self.variables = variables

        graph_data = bic_exact_search(self.data)
        self.adjancency_matrix = graph_data[0]
        edges = self.get_directed_edges()

        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(range(len(self.adjancency_matrix)))
        self.graph.add_edges_from(edges)

        # Rename nodes
        mapping = {}
        for i in range(len(self.variables)):
            mapping[i] = self.variables[i]
        self.graph = nx.relabel_nodes(self.graph, mapping)

##############################################
############## GRAPH FUNCTIONS ###############
##############################################

    def get_directed_edges(self):
        edges = []
        for i in range(len(self.adjancency_matrix)):
            for j in range(len(self.adjancency_matrix)):
                if self.adjancency_matrix[i][j] == 1 and self.adjancency_matrix[j][i] == 0:
                    edges.append((i, j))
        return edges

    def has_relations(self, node):
        return len(list(self.graph.predecessors(node))) > 0 or len(list(self.graph.successors(node))) > 0

    def get_nodes(self):
        return list(self.graph.nodes())
        
    def get_edges(self):
        return list(self.graph.edges())

    def get_graph(self):
        return self.graph