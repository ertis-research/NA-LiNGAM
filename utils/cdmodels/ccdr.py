##############################################
############## IMPORT LIBRARIES ##############
##############################################
from cdt.causality.graph import CCDr
import networkx as nx
import numpy as np

##############################################
############## CLASS DEFINITION ##############
##############################################

class CCDrAlgorithm():
    def __init__(self, data, variables):
        self.data = data
        self.variables = variables

        cd = CCDr()
        self.graph = cd.predict(self.data)
                
        self.nodes = list(self.graph.nodes())
        self.edges = list(self.graph.edges())

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

    ccdr = CCDrAlgorithm(dataframe, variables)
    print(ccdr.get_edges())
    print(ccdr.get_nodes())