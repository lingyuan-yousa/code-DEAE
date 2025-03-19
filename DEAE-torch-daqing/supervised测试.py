import random

import numpy as np
import warnings

warnings.filterwarnings("ignore")

from data_loader import load_mnist_data
from supervised_models_mlp import MLP, train_mlp_pytorch, predict_mlp_pytorch
from supervised_models_xgboost import xgb_model
from supervised_models_logit import logit

from deae_utils import perf_metric

# Experimental parameters
model_sets = ['logit', 'xgboost', 'mlp']

label_data_rate = 1

# Metric
metric = 'acc'

# Define output
results = np.zeros([len(model_sets) + 2])


x_train, y_train, x_unlab, x_test, y_test = load_mnist_data(label_data_rate)

# Logistic regression
y_test_hat = logit(x_train, y_train, x_test)
results[0] = perf_metric(metric, y_test, y_test_hat)

# XGBoost
y_test_hat = xgb_model(x_train, y_train, x_test)
results[1] = perf_metric(metric, y_test, y_test_hat)

# for ss in range(1, 100):
# x_train, y_train, x_unlab, x_test, y_test = load_mnist_data(label_data_rate, ss)
# print('seed' + str(ss))

# MLP
# 定义模型参数
input_dim = x_train.shape[1]
hidden_dim = 64  # 示例隐藏层维度
unique_labels = np.unique(y_train)
output_dim = len(unique_labels)
activation_fn = 'relu'

# 创建模型实例
model = MLP(input_dim, hidden_dim, output_dim, activation_fn)

# 定义训练参数
mlp_parameters = {
    'batch_size': 128,
    'epochs': 100,
    'lr': 0.002,
}

# 训练模型
train_mlp_pytorch(x_train, y_train, model, mlp_parameters, x_test, y_test)
# 预测
y_test_hat = predict_mlp_pytorch(x_test, model)
results[2] = perf_metric(metric, y_test, y_test_hat)


# Report performance
for m_it in range(len(model_sets)):
    model_name = model_sets[m_it]
    print('Supervised Performance, Model Name: ' + model_name +
          ', Performance: ' + str(results[m_it]))
