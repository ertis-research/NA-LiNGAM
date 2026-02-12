import numpy as np
import pandas as pd
import random

import networkx as nx
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from causallearn.utils.Dataset import load_dataset
from utils.constants import adabyron_simplified_vars, adabyron_posible_edges, adabyron_renames

class AdaByronData:
    def __init__(self, n_noise=0, diff=True):
        """
        Creates a real data graph based on the AdaByron dataset with optional noise nodes.

        Parameters:
        - n_noise (int): Number of noise nodes to add to the dataset.
        - diff (bool): Whether to use differenced data or not.
        """

        if diff:
            df = pd.read_csv('datasets/adabyron_preprocessed_hourly_residuals_diffArima.csv')
        else:
            df = pd.read_csv('datasets/adabyron_preprocessed_hourly.csv')
        df.dropna(inplace=True)

        self.dataframe = df.copy()

        self.real_nodes = adabyron_simplified_vars
        
        # Extra data processing specific to AdaByron dataset
        self.dataframe['_time'] = pd.to_datetime(self.dataframe['_time'], format='mixed')
        self.dataframe = self.dataframe.set_index('_time')
        self.dataframe = self.dataframe[self.real_nodes]
        self.dataframe = self.dataframe.resample('H').mean()
        self.dataframe.dropna(inplace=True)

        for j in range(n_noise):
            np.random.seed(None)
            shift_amount = np.random.randint(low=12, high=24)
            # print('Noise node', j, 'shift amount:', shift_amount)
            original_var = random.choice(self.real_nodes)
            shifted_signal = np.roll(self.dataframe[original_var].values, shift_amount)
            self.dataframe[f'Noise_{j}'] = shifted_signal

        real_edges = adabyron_posible_edges

        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.real_nodes)
        self.graph.add_edges_from(real_edges)

    def get_nodes(self):
        """
        Returns the list of nodes in the graph.
        """
        return list(self.dataframe.columns)
    
    def get_edges(self):
        """
        Returns the list of edges in the real graph.
        """
        return list(self.graph.edges())
    
    def get_dataframe(self):
        """
        Returns the dataframe used to create the graph.
        """
        return self.dataframe
    
    def get_graph(self):
        """
        Returns the real graph created from the dataframe.
        """
        return self.graph

class RealDataSachs:
    def __init__(self, n_noise=0):
        """
        Creates a real data graph based on the Sachs dataset with optional noise nodes.

        Parameters:
        - n_noise (int): Number of noise nodes to add to the dataset.
        """
        data, labels = load_dataset(dataset_name="sachs")
        self.dataframe = pd.DataFrame(data=data,columns=labels)

        self.dataframe.rename(columns={
            'raf': 'Raf',
            'mek': 'Mek',
            'plc' : 'Plcg',
            'pip2' : 'PIP2',
            'pip3' : 'PIP3',
            'erk' : 'Erk',
            'akt' : 'Akt',
            'pka' : 'PKA',
            'pkc' : 'PKC',
            'p38' : 'P38',
            'jnk' : 'Jnk',
            }
            , inplace=True)
        
        self.dataframe.drop(columns=['Plcg', 'PIP2', 'PIP3'], inplace=True)

        for col in self.dataframe.columns:
            mean = self.dataframe[col].mean()
            std = self.dataframe[col].std()
            self.dataframe = self.dataframe[(self.dataframe[col] >= mean - std) & (self.dataframe[col] <= mean + std)]
        self.dataframe = self.dataframe.reset_index(drop=True)

        self.real_nodes = list(self.dataframe.columns)

        for j in range(n_noise):
            self.dataframe[f'Noise_{j}'] =  np.sin(2 * np.pi * np.arange(len(self.dataframe)) / 50) + np.random.normal(0, 0.1, len(self.dataframe))

        scaler = StandardScaler()
        self.dataframe[self.dataframe.columns] = scaler.fit_transform(self.dataframe[self.dataframe.columns])
        self.dataframe = self.dataframe.dropna()
        self.dataframe = self.dataframe.reset_index(drop=True)

        real_edges = [
            ('PKC', 'Mek'),
            ('PKC', 'Raf'),
            ('PKC', 'PKA'),
            ('PKC', 'Jnk'),
            ('PKC', 'P38'),
            ('PKA', 'Raf'),
            ('PKA', 'Mek'),
            ('PKA', 'Erk'),
            ('PKA', 'Akt'),
            ('PKA', 'Jnk'),
            ('PKA', 'P38'),
            ('Raf', 'Mek'),
            ('Mek', 'Erk'),
            ('Erk', 'Akt')
        ]

        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.real_nodes)
        self.graph.add_edges_from(real_edges)

    def get_nodes(self):
        """
        Returns the list of nodes in the graph.
        """
        return list(self.dataframe.columns)
    
    def get_edges(self):
        """
        Returns the list of edges in the real graph.
        """
        return list(self.graph.edges())
    
    def get_dataframe(self):
        """
        Returns the dataframe used to create the graph.
        """
        return self.dataframe
    
    def get_graph(self):
        """
        Returns the real graph created from the dataframe.
        """
        return self.graph