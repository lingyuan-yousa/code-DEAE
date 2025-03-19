import numpy as np
import os
import warnings

import torch

warnings.filterwarnings("ignore")

from data_loader import load_mnist_data
from supervised_models_mlp import MLP, train_mlp_pytorch, predict_mlp_pytorch
from supervised_models_xgboost import xgb_model
from supervised_models_logit import logit

from deae_self import VIME_Self
from deae_semi import train_model
from deae_utils import perf_metric

# Experimental parameters
label_no = 1000
model_sets = ['logit', 'xgboost', 'mlp']

# Hyper-parameters
p_m = 0.3
alpha = 2.0
K = 3
beta = 1.0
label_data_rate = 0.3
lr = 0.005

# Metric
metric = 'acc'

# Define output
results = np.zeros([len(model_sets) + 2])

x_train, y_train, x_unlab, x_test, y_test = load_mnist_data(label_data_rate)

# Use subset of labeled data
# x_train = x_train[:label_no, :]
# y_train = y_train[:label_no, :]

# Logistic regression
y_test_hat = logit(x_train, y_train, x_test)
results[0] = perf_metric(metric, y_test, y_test_hat)

# XGBoost
y_test_hat = xgb_model(x_train, y_train, x_test)
results[1] = perf_metric(metric, y_test, y_test_hat)

# MLP
# 定义模型参数
input_dim = x_train.shape[1]
hidden_dim = 64  # 示例隐藏层维度
output_dim = y_train.shape[1] # 假设y_train是类别索引
activation_fn = 'relu'

# 创建模型实例
model = MLP(input_dim, hidden_dim, output_dim, activation_fn)

# 定义训练参数
mlp_parameters = {
    'batch_size': 128,
    'epochs': 200,
    'lr': 0.008,
}

# 训练模型
train_mlp_pytorch(x_train, y_train, model, mlp_parameters)
# 预测
y_test_hat = predict_mlp_pytorch(x_test, model)
results[2] = perf_metric(metric, y_test, y_test_hat)

# Report performance
for m_it in range(len(model_sets)):
    model_name = model_sets[m_it]
    print('Supervised Performance, Model Name: ' + model_name +
          ', Performance: ' + str(results[m_it]))

# Train VIME-Self
vime_self_parameters = dict()
vime_self_parameters['batch_size'] = 128
vime_self_parameters['epochs'] = 100

file_name = './save_model/encoder_model'


_, dim = x_unlab.shape
vime_self_model = VIME_Self(dim, alpha)
vime_self_model.train_model(x_unlab, p_m, vime_self_parameters)

# Save encoder
if not os.path.exists('save_model'):
    os.makedirs('save_model')
# 假设 model 是你的 VIME_Self 模型实例
torch.save(vime_self_model.encoder.state_dict(), file_name)

# 首先创建一个与保存的编码器结构相同的新编码器实例
encoder = VIME_Self(dim, alpha).encoder  # 假设 dim 和 alpha 已经定义

# 加载保存的 state_dict
encoder.load_state_dict(torch.load(file_name))
encoder.eval()  # 切换到评估模式

with torch.no_grad():  # 禁用梯度计算
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)

    x_train_hat = encoder(x_train_tensor)
    x_test_hat = encoder(x_test_tensor)

    y_test_hat = predict_mlp_pytorch(x_test_hat, model)

    results[3] = perf_metric(metric, y_test, y_test_hat)

print('VIME-Self Performance: ' + str(results[3]))

# Train VIME-Semi
vime_semi_parameters = dict()
vime_semi_parameters['hidden_dim'] = 16
vime_semi_parameters['batch_size'] = 128
vime_semi_parameters['iterations'] = 50
y_test_hat = train_model(encoder, x_train, y_train, x_unlab, x_test,
                       vime_semi_parameters, p_m, K, beta, lr)

# Test VIME
results[4] = perf_metric(metric, y_test, y_test_hat)

print('VIME Performance: ' + str(results[4]))