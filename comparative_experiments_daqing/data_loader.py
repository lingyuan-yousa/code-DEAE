# Necessary packages
import random

import numpy as np
import pandas as pd
import torch
import os
from sklearn import preprocessing


def load_mnist_data(label_data_rate):

    seed = 37
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    current_dir = os.path.dirname(os.path.abspath(__file__))  # /code-DEAE/comparative_experiments_daqing/

    project_root = os.path.dirname(current_dir)  # /code-DEAE/
    data_path = os.path.join(project_root, 'data', 'daqing1.csv')

    df = pd.read_csv(data_path, encoding='utf-8-sig')

    max_min = preprocessing.StandardScaler()

    # Split the dataset into training and test sets
    df_train = df[~df['Well_Name'].str.contains('Le')]
    df_test = df[df['Well_Name'].str.contains('Le')]

    # Select specific columns as features
    x_train = df_train.iloc[:, 3:16]
    x_train = max_min.fit_transform(x_train)
    y_train = df_train['LITH']

    x_test = df_test.iloc[:, 3:16]
    x_test = max_min.transform(x_test)  # Should use the same scaler object to transform test data
    y_test = df_test['LITH']

    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train.to_numpy(), dtype=torch.long)
    y_test = torch.tensor(y_test.to_numpy(), dtype=torch.long)

    y_train = y_train - 1
    y_test = y_test - 1

    idx = torch.randperm(x_train.size(0))

    label_idx = idx[:int(len(idx) * label_data_rate)]
    unlab_idx = idx[int(len(idx) * label_data_rate):]

    x_label = x_train[label_idx, :]
    y_label = y_train[label_idx]

    x_unlab = x_train[unlab_idx, :]

    return x_label, y_label, x_unlab, x_test, y_test