##############################################
############## IMPORT LIBRARIES ##############
##############################################

from ylearn.causal_discovery import CausalDiscovery
import networkx as nx
import numpy as np

##############################################
############## CLASS DEFINITION ##############
##############################################

class NOTEARSAlgorithm():
    def __init__(self, data, variables):
        self.data = data
        self.variables = variables

        cd = CausalDiscovery(hidden_layer_dim=[3])
        est_dict = cd(self.data, threshold=0.01, return_dict=True)
        print(est_dict)
        # OrderedDict([('A', []), ('B', ['A', 'C', 'D', 'E']), ('C', []), ('D', ['A', 'E']), ('E', [])])
        self.nodes = list(est_dict.keys())
        self.edges = []
        for child, parents in est_dict.items():
            for parent in parents:
                self.edges.append((parent, child))
                
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.nodes)
        self.graph.add_edges_from(self.edges)


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
    
if __name__ == '__main__':
    data = np.random.rand(100, 5)
    variables = ['A', 'B', 'C', 'D', 'E']

    import pandas as pd
    dataframe = pd.DataFrame(data, columns=variables)

    notears = NOTEARSAlgorithm(dataframe, variables)
    print(notears.get_edges())
    print(notears.get_nodes())