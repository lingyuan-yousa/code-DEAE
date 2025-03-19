import random

import numpy as np
import warnings

import torch
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score


warnings.filterwarnings("ignore")

from data_loader import load_mnist_data
from supervised_models_mlp import MLP, train_mlp_pytorch, predict_mlp_pytorch
from RDAE_encoder import RecurrentDenoisingAutoencoder
from PL_1DCNN_MLP import train_model
from vime_utils import perf_metric

def set_seed(seed):
    """设置随机种子以确保可重复性"""
    torch.manual_seed(seed)  # 为CPU设置随机种子
    np.random.seed(seed)  # Numpy模块的随机种子
    random.seed(seed)  # Python内置的随机模块
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # 如果网络输入数据维度或类型上变化不大，设置True可能会增加运行效率

model_sets = []

# Hyper-parameters
p_m = 0.3
alpha = 2
K = 3
beta = 0.1
label_data_rate = 0.2
# Metric
metric = 'acc'

# Define output
results = np.zeros([len(model_sets) + 2])
x_train, y_train, x_unlab, x_test, y_test = load_mnist_data(label_data_rate)
file_name = './save_model/encoder_model'
_, dim = x_unlab.shape

# 首先创建一个与保存的编码器结构相同的新编码器实例
encoder = RecurrentDenoisingAutoencoder(dim).encoder  # 假设 dim 和 alpha 已经定义

# MLP
input_dim = x_train.shape[1]
hidden_dim = 64  # 示例隐藏层维度
unique_labels = np.unique(y_train)
output_dim = len(unique_labels)
activation_fn = 'relu'

seed = 10

set_seed(seed)
model = MLP(input_dim, hidden_dim, output_dim, activation_fn)

mlp_parameters = {
    'batch_size': 128,
    'epochs': 100,
    'lr': 0.002,
}

# 加载保存的 state_dict
encoder.load_state_dict(torch.load(file_name))
encoder.eval()  # 切换到评估模式

with torch.no_grad():
    x_train_hat = encoder(x_train)
    x_test_hat = encoder(x_test)
    x_train_hat, (_, _) = encoder(x_train)
    x_test_hat, (_, _) = encoder(x_test)


train_mlp_pytorch(x_train_hat, y_train, model, mlp_parameters)
y_test_hat = predict_mlp_pytorch(x_test_hat, model)


results[0] = perf_metric(metric, y_test, y_test_hat)

print('VIME-Self Performance: ' + str(results[0]))

# Train VIME-Semi
vime_semi_parameters = dict()
vime_semi_parameters['hidden_dim'] = 64
vime_semi_parameters['batch_size'] = 128
vime_semi_parameters['iterations'] = 56
vime_semi_parameters['lr'] = 0.005

# for seed in range(20, 50):
#     print('seed ' + str(seed))

set_seed(14)
y_test_hat = train_model(encoder, x_train, y_train, x_unlab, x_test, y_test,
                       vime_semi_parameters, p_m, K, beta)

# 计算准确率
accuracy = accuracy_score(y_test, y_test_hat)
# 计算精确率
precision = precision_score(y_test, y_test_hat, average='weighted')
# 计算召回率
recall = recall_score(y_test, y_test_hat, average='weighted')
# 计算 F1 分数
f1_score_value = f1_score(y_test, y_test_hat, average='weighted')

print(f'准确率：{accuracy:.4f}')
print(f'精确率：{precision:.4f}')
print(f'召回率：{recall:.4f}')
print(f'F1 分数：{f1_score_value:.4f}')

report = classification_report(y_test, y_test_hat)
print(report)