##############################################
############## IMPORT LIBRARIES ##############
##############################################

from causallearn.search.Granger.Granger import Granger
from causallearn.utils.GraphUtils import GraphUtils
import networkx as nx
import numpy as np
import pydotplus

##############################################
############## CLASS DEFINITION ##############
##############################################

class GrangerAlgorithm():
    def __init__(self, data, variables):
        self.data = np.array(data[variables])
        self.variables = variables

        granger = Granger()
        lasso = granger.granger_lasso(self.data)

        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(range(len(variables)))
        self.graph.add_edges_from([(i, j) for i in range(len(variables)) for j in range(len(variables)) if lasso[i][j] != 0])

        # Rename nodes
        mapping = {}
        for i in range(len(self.variables)):
            mapping[i] = self.variables[i]
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

if __name__ == '__main__':
    data = np.random.rand(100, 5)
    variables = ['A', 'B', 'C', 'D', 'E']

    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import io
    dataframe = pd.DataFrame(data, columns=variables)

    pc = GrangerAlgorithm(dataframe, variables)
    print(pc.get_nodes())
    print(pc.get_edges())
    print(pc.get_graph())

    # pyd = GraphUtils.to_pydot(pc.algorithm_graph)
    # tmp_png = pyd.create_png(f="png")
    # fp = io.BytesIO(tmp_png)
    # img = mpimg.imread(fp, format='png')
    # plt.axis('off')
    # plt.imshow(img)
    # plt.savefig('test_algorithm_graph.png')