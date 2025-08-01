import numpy as np
import pandas as pd
import random

import networkx as nx
from sklearn.preprocessing import MinMaxScaler

class SyntheticGraphGenerator:
    def __init__(self, sample_size, node_size, min_edges=1, n_noise_nodes=0, max_edges=None, seed=None):
        if n_noise_nodes > node_size:
            raise ValueError("Number of noise nodes must be less than or equal to the number of nodes.")
        if min_edges < 1:
            raise ValueError("Minimum number of edges must be at least 1.")
        if max_edges is not None and max_edges < 1:
            raise ValueError("Maximum number of edges must be at least 1.")
        if max_edges is not None and max_edges < min_edges:
            raise ValueError("Maximum number of edges must be greater than or equal to the minimum number of edges.")
        if node_size > node_size:
            raise ValueError("Number of nodes must be greater than or equal to the number of noise nodes.")

        self.sample_size = sample_size
        self.node_size = node_size
        self.min_edges = min_edges
        self.n_noise_nodes = n_noise_nodes
        self.seed = seed

        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)

        if max_edges is None:
            self.max_edges = node_size - 1
        else:
            self.max_edges = max_edges

        self.dataframe = None
        self.graph = None
        self.adjanecy_matrix = None
        self.ponderated_matrix = None
        self.noise_nodes = None
        self.real_nodes = None

    def generate_samples(self, scale_data=False):
        data_list = []
        noise_nodes = []
        self.adjanecy_matrix = np.zeros((self.node_size, self.node_size))
        self.ponderated_matrix = np.zeros((self.node_size, self.node_size))
        

        for i in range(self.node_size - self.n_noise_nodes):
            node_sample = np.random.random(self.sample_size)
            edge_count = random.randint(min(self.min_edges, i), min(self.max_edges, i))

            while np.count_nonzero(self.adjanecy_matrix[i]) < edge_count:
                possible_edges = [j for j in range(i) if self.adjanecy_matrix[i][j] == 0]
                if len(possible_edges) == 0:
                    print("No possible edges.")
                    break # No debería pasar
                random_choice = random.choice(possible_edges)
                self.adjanecy_matrix[i][random_choice] = 1

            for j in range(i):
                if self.adjanecy_matrix[i][j] == 1:
                    rand_pond = np.random.random()
                    node_sample += data_list[j] * rand_pond
                    self.ponderated_matrix[i][j] = rand_pond

            node_sample = np.array(node_sample)
            data_list.append(node_sample)

        for _ in range(self.n_noise_nodes):
            noise_nodes.append(len(data_list))
            data_list.append(np.random.random(self.sample_size))

        self.adjanecy_matrix = self.adjanecy_matrix.T
        self.ponderated_matrix = self.ponderated_matrix.T

        self.dataframe = pd.DataFrame(data_list).T
        self.dataframe.columns = [chr(i+65) for i in range(len(self.dataframe.columns))]
        # Scale data
        
        scaler = MinMaxScaler()
        
        if scale_data:
            self.dataframe = pd.DataFrame(scaler.fit_transform(self.dataframe), columns=self.dataframe.columns, index=self.dataframe.index)

        self.noise_nodes = list(self.dataframe.columns[noise_nodes])
        self.real_nodes = list(self.dataframe.columns[[i for i in range(self.node_size) if i not in noise_nodes]])

        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.real_nodes)
        for i in range(self.node_size):
            for j in range(self.node_size):
                if self.adjanecy_matrix[i][j] == 1:
                    self.graph.add_edge(self.dataframe.columns[i], self.dataframe.columns[j])

        # print("Samples generated.")

    def get_nodes(self):
        if self.dataframe is None:
            return None
        return list(self.dataframe.columns)
    
    def get_real_noise_nodes(self):
        return self.real_nodes, self.noise_nodes
    
    def get_edges(self):
        if self.graph is None:
            return None
        return list(self.graph.edges())
    
    def get_dataframe(self):
        return self.dataframe
    
    def get_graph(self):
        return self.graph