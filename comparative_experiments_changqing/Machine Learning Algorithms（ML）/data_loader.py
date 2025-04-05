import os

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def load_data(label_data_rate):
    seed = 15
    np.random.seed(seed)

    current_dir = os.path.dirname(
        os.path.abspath(__file__))  # cur dir：/code-DEAE/comparative_experiments_daqing/Machine Learning Algorithms（ML）/

    project_root = os.path.dirname(os.path.dirname(current_dir))  # /code-DEAE/
    data_path = os.path.join(project_root, 'data', 'changqing1.csv')

    df = pd.read_csv(data_path, encoding='utf-8-sig')
    max_min = preprocessing.StandardScaler()

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    # Select specific columns as features
    x_train = df_train.iloc[:, 2:13].values
    x_train = max_min.fit_transform(x_train)
    y_train = df_train['LITH'].values

    x_test = df_test.iloc[:, 2:13].values
    x_test = max_min.transform(x_test)  # Use the same scaler object to transform test data
    y_test = df_test['LITH'].values

    y_train -= 1  # Subtract 1 from labels to start encoding from 0
    y_test -= 1

    idx = np.random.permutation(x_train.shape[0])

    label_idx = idx[:int(len(idx) * label_data_rate)]
    unlab_idx = idx[int(len(idx) * label_data_rate):]

    x_label = x_train[label_idx, :]
    y_label = y_train[label_idx]

    x_unlab = x_train[unlab_idx, :]

    return x_label, y_label, x_unlab, x_test, y_test