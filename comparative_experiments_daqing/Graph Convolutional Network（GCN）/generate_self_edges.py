import os
import sys
import numpy as np
import pandas as pd

# Add parent directory to Python path for imports
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from data_loader import load_mnist_data

def generate_self_edges(data_length):
    """Generate self-loop edges for each node."""
    edges = [[i, i] for i in range(data_length)]
    return edges

def main():
    # Load data using the correct train/test split
    x_label, y_label, x_unlab, x_test, y_test = load_mnist_data(0.3)
    
    # Generate edges for training and test sets
    train_edges = generate_self_edges(len(x_label) + len(x_unlab))  # Full training set
    test_edges = generate_self_edges(len(x_test))
    
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