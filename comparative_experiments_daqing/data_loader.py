import random
import numpy as np
import pandas as pd
import torch
import os
from sklearn import preprocessing

def load_mnist_data(label_data_rate):
    """Load data with well-based train/test split.
    
    Args:
        label_data_rate: Proportion of labeled data in training set
        
    Returns:
        x_label: Labeled training features
        y_label: Labeled training labels
        x_unlab: Unlabeled training features
        x_test: Test features
        y_test: Test labels
    """
    seed = 37
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Load data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, 'data', 'daqing1.csv')
    
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    
    # Split by well name
    df_train = df[~df['Well_Name'].str.contains('Le')]
    df_test = df[df['Well_Name'].str.contains('Le')]
    
    # Standardize features
    max_min = preprocessing.StandardScaler()
    
    # Extract and transform training features
    x_train = df_train.iloc[:, 3:16].values
    x_train = max_min.fit_transform(x_train)
    y_train = df_train['LITH'].values - 1  # Convert to 0-based indexing
    
    # Extract and transform test features
    x_test = df_test.iloc[:, 3:16].values
    x_test = max_min.transform(x_test)  # Use same scaler as training
    y_test = df_test['LITH'].values - 1  # Convert to 0-based indexing
    
    # Split training data into labeled and unlabeled
    indices = np.random.permutation(len(x_train))
    n_labeled = int(len(indices) * label_data_rate)
    
    labeled_idx = indices[:n_labeled]
    unlabeled_idx = indices[n_labeled:]
    
    x_label = x_train[labeled_idx]
    y_label = y_train[labeled_idx]
    x_unlab = x_train[unlabeled_idx]
    
    return x_label, y_label, x_unlab, x_test, y_test

def get_well_names():
    """Get training and test well names for reference."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, 'data', 'daqing1.csv')
    
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    train_wells = df[~df['Well_Name'].str.contains('Le')]['Well_Name'].unique()
    test_wells = df[df['Well_Name'].str.contains('Le')]['Well_Name'].unique()
    
    return train_wells, test_wells