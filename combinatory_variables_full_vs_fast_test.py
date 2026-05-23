import numpy as np
import pandas as pd
import random

import networkx as nx
from sklearn.preprocessing import MinMaxScaler

from utils.nalingam_environment import GraphEnvNALiNGAM

# Create a synthetic dataset with this graph structure: [A->B, B->C, A->C, C->D, E->D, F->D, E->F]
# A, B, C are part of the initial graph, and the other ones are aditional ones.

def create_synthetic_dataset(sample_size=1000, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Define the graph structure
    real_graph = nx.DiGraph()
    real_graph.add_edges_from([('A', 'B'), ('B', 'C'), ('A', 'C'), ('C', 'D'), ('E', 'D'), ('F', 'D'), ('E', 'F')])

    initial_graph = nx.DiGraph()
    initial_graph.add_edges_from([('A', 'B'), ('B', 'C'), ('A', 'C')])

    # Generate synthetic data for each node
    A = np.random.rand(sample_size)
    B = 0.22 * A + 0.32 * np.random.rand(sample_size)
    C = 0.1 * A + 0.9 * B + 0.3 * np.random.rand(sample_size)
    E = np.random.rand(sample_size)
    F = 0.7 * E + 0.2 * np.random.rand(sample_size)
    D = 0.7 * C + 0.2 * E + 0.1 * F + 0.20 * np.random.rand(sample_size)

    # Create a DataFrame from the generated data
    df = pd.DataFrame({'A': A, 'B': B, 'C': C, 'D': D, 'E': E, 'F': F})

    return df, initial_graph, real_graph

df, initial_graph, real_graph = create_synthetic_dataset(sample_size=1000, seed=42)

env = GraphEnvNALiNGAM(df, initial_graph)
optimal_state = np.ones(len(env.new_features))

calculated_state, _ = env.get_best_state_fast(30)
calculated_state_slow, _ = env.get_best_state(30)

print(f"Calculated state: {calculated_state}"
      f"\nOptimal state: {optimal_state}")
print(f"Calculated state slow: {calculated_state_slow}"
      f"\nOptimal state: {optimal_state}")