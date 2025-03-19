# Necessary packages
import random

import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split



def load_mnist_data(label_data_rate):

    seed = 37
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 分割训练集和测试集
    # df_train = df[~df['Well_Name'].str.contains('里')]
    # df_test = df[df['Well_Name'].str.contains('里')]

    df = pd.read_csv('changqing.csv', encoding='utf-8')  # 或使用你知道的正确编码
    max_min = preprocessing.StandardScaler()

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    # 选择特定列作为特征
    x_train = df_train.iloc[:, 2:13]
    x_train = max_min.fit_transform(x_train)
    y_train = df_train['LITH']

    x_test = df_test.iloc[:, 2:13]
    x_test = max_min.transform(x_test)  # 应使用相同的scaler对象来转换测试数据
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