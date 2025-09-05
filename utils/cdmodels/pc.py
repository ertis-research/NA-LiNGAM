##############################################
############## IMPORT LIBRARIES ##############
##############################################

from causallearn.search.ConstraintBased.PC import pc
import networkx as nx
import numpy as np

##############################################
############## CLASS DEFINITION ##############
##############################################

class PCAlgorithm():
    def __init__(self, data, variables):
        self.data = np.array(data[variables])
        self.variables = variables

        initial_graph = pc(self.data).G

        node_list = []
        for node in initial_graph.nodes:
            node_list.append(node.get_name())

        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(node_list)
        self.graph.add_edges_from([(i.get_name(), j.get_name()) for i in initial_graph.nodes for j in initial_graph.nodes if initial_graph.is_directed_from_to(i, j)])

        # Rename nodes
        mapping = {}
        for i in range(len(self.variables)):
            mapping[node_list[i]] = self.variables[i]
        self.graph = nx.relabel_nodes(self.graph, mapping)

##############################################
############## GRAPH FUNCTIONS ###############
##############################################

    def has_relations(self, node):
        return len(list(self.graph.predecessors(node))) > 0 or len(list(self.graph.successors(node))) > 0

    def get_nodes(self):
        return list(self.graph.nodes())
        
    def get_edges(self):
        return list(self.graph.edges())

    def get_graph(self):
        return self.graph