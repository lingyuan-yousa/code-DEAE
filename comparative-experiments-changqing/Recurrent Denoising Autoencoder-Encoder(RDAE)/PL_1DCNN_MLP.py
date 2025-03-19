import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from vime_utils import mask_generator, pretext_generator  # 确保这些函数与 PyTorch 兼容

from itertools import cycle

# class PredictorModel(nn.Module):
#     def __init__(self, data_dim, hidden_dim, label_dim):
#         super(PredictorModel, self).__init__()
#         self.network = nn.Sequential(
#             nn.Linear(data_dim, hidden_dim),
#             # nn.Dropout(p=0.01),
#             nn.LeakyReLU(),
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             # nn.Dropout(p=0.01),
#             nn.LeakyReLU(),
#             nn.Linear(hidden_dim // 2, label_dim)
#         )
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         y_hat_logit = self.network(x)
#         y_hat = self.softmax(y_hat_logit)
#         return y_hat_logit, y_hat

class PredictorModel(nn.Module):
    def __init__(self, data_dim, hidden_dim, label_dim):
        super(PredictorModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),  # 假设data_dim足够大
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * (data_dim // 4), hidden_dim),  # 需要根据CNN输出调整
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, label_dim),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.unsqueeze(1)  # 增加通道维度
        x = self.cnn(x)
        x = x.view(x.size(0), -1)  # 展平
        y_hat_logit = self.fc(x)
        y_hat = self.softmax(y_hat_logit)
        return y_hat_logit, y_hat

def set_seed(seed):
    """设置随机种子以确保可重复性"""
    torch.manual_seed(seed)  # 为CPU设置随机种子
    np.random.seed(seed)  # Numpy模块的随机种子
    random.seed(seed)  # Python内置的随机模块
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # 如果网络输入数据维度或类型上变化不大，设置True可能会增加运行效率


def train_model(encoder, x_train, y_train, x_unlab, x_test, y_test, parameters, p_m, K, beta, T1=20, T2=100):
    hidden_dim = parameters['hidden_dim']

    # Basic parameters
    data_dim = len(x_train[0, :])
    label_dim = len(np.unique(y_train))

    # 训练设置
    predictor = PredictorModel(data_dim, hidden_dim, label_dim)
    optimizer = optim.Adam(predictor.parameters(), lr=parameters['lr'], betas=(0.9, 0.999), weight_decay=1e-6, amsgrad=False)
    supervised_loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        x_test, (_, _) = encoder(x_test)

    y_pseudo = torch.zeros(len(x_unlab), dtype=torch.long)  # 假设类别从0开始

    alpha = 0.0
    for epoch in range(parameters['iterations']):
        predictor.train()  # 切换到训练模式

        optimizer.zero_grad()  # 梯度清零

        with torch.no_grad():
            x_encoded, (_, _) = encoder(x_train)  # 编码有标签数据

        unsupervised_loss = 0
        for idx in range(K):
            with torch.no_grad():
                xu_temp, (_, _) = encoder(x_unlab)  # 编码无标签数据

            yv_hat_logit, yv_hat = predictor(xu_temp)
            unsupervised_loss += supervised_loss_fn(yv_hat_logit, y_pseudo)
            _, y_pseudo = torch.max(yv_hat_logit, dim=1)

        unsupervised_loss /= K

        # 计算有监督损失
        y_hat_logit, y_hat = predictor(x_encoded)
        supervised_loss = supervised_loss_fn(y_hat_logit, y_train)

        if epoch > T1:
            alpha = (epoch - T1) / (T2 - T1) * beta
            if epoch > T2:
                alpha = beta


        total_loss = supervised_loss + alpha * unsupervised_loss
        total_loss.backward()  # 反向传播
        optimizer.step()  # 参数更新


        test_accuracy = compute_accuracy(predictor, x_test, y_test)

        print(f'Epoch {epoch + 1}, Supervised Loss: {supervised_loss.item():.4f}, '
              f'Unsupervised Loss: {unsupervised_loss.item():.4f}, Total Loss: {total_loss:.4f}, '
              f'Test Accuracy: {test_accuracy:.2f}%')


    # predictor.load_state_dict(torch.load(class_file_name))
    predictor.eval()  # 切换到评估模式

    # 在PyTorch中，进行预测时通常需要禁用梯度计算
    with torch.no_grad():
        y_test_hat_logit, y_test_hat = predictor(x_test)

    # 如果需要，转换y_test_hat为NumPy数组
    y_test_hat = np.argmax(y_test_hat.numpy(), axis=1)

    return y_test_hat


def compute_accuracy(predictor, x_test, y_test):
    predictor.eval()  # 切换到评估模式
    with torch.no_grad():  # 禁用梯度计算
        # x_test_encoded = encoder(x_test)  # 对测试集进行编码
        y_test_pred_logit, y_test_pred = predictor(x_test)  # 对测试集进行预测
        _, y_test_pred = torch.max(y_test_pred_logit, dim=1)

        correct = (y_test_pred == y_test).sum().item()  # 计算正确预测的数量
        total = y_test.size(0)  # 测试集的总数
        accuracy = 100 * correct / total  # 计算准确率
    return accuracy


