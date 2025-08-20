import os
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

def generate_edges(data, n_neighbors=5):
    """Generate edges based on k-nearest neighbors in feature space."""
    # Use features for neighbor calculation
    features = data.iloc[:, 3:16].values
    
    # Find k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(features)  # +1 because point itself is included
    distances, indices = nbrs.kneighbors(features)
    
    # Generate edges (skip first neighbor as it's the point itself)
    edges = []
    for i in range(len(data)):
        for j in indices[i][1:]:  # Skip first neighbor (self)
            edges.append([i, j])
    
    return edges

def main():
    # Read data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
    train_data = pd.read_csv(os.path.join(data_dir, "daqing.csv"))
    test_data = pd.read_csv(os.path.join(data_dir, "daqing1.csv"))
    
    # Generate edges
    train_edges = generate_edges(train_data)
    test_edges = generate_edges(test_data)
    
    # Save edges
    edges_dir = os.path.join(os.path.dirname(__file__), 'edges')
    os.makedirs(edges_dir, exist_ok=True)
    
    # Save train edges
    with open(os.path.join(edges_dir, 'daqing_train_edges.txt'), 'w') as f:
        for edge in train_edges:
            f.write(f"{edge[0]} {edge[1]}\n")
    
    # Save test edges
    with open(os.path.join(edges_dir, 'daqing_test_edges.txt'), 'w') as f:
        for edge in test_edges:
            f.write(f"{edge[0]} {edge[1]}\n")

if __name__ == '__main__':
    main()
