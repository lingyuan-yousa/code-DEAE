import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from deae_utils import mask_generator, pretext_generator  # 确保这些函数与 PyTorch 兼容

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


def unsupervised_loss_fn(yv_hat_logit):
    # 计算沿指定维度（这里是每个特征across the batch）的方差
    variance = torch.var(yv_hat_logit, dim=0, unbiased=False)
    # 计算方差的均值作为无监督损失
    return torch.mean(variance)

def set_seed(seed):
    """设置随机种子以确保可重复性"""
    torch.manual_seed(seed)  # 为CPU设置随机种子
    np.random.seed(seed)  # Numpy模块的随机种子
    random.seed(seed)  # Python内置的随机模块
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # 如果网络输入数据维度或类型上变化不大，设置True可能会增加运行效率


# def train_model(encoder, x_train, y_train, x_unlab, x_test, y_test, parameters, p_m, K, beta, seed):
def train_model(encoder, x_train, y_train, x_unlab, x_test, y_test, parameters, p_m, K, beta):
    hidden_dim = parameters['hidden_dim']

    # Basic parameters
    data_dim = len(x_train[0, :])
    label_dim = len(np.unique(y_train))

    # 训练设置
    predictor = PredictorModel(data_dim, hidden_dim, label_dim)
    optimizer = optim.Adam(predictor.parameters(), lr=parameters['lr'], weight_decay=0, amsgrad=False)
    supervised_loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        x_test = encoder(x_test)

    # 数据准备
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=parameters['batch_size'], shuffle=True)

    unlab_dataset = TensorDataset(x_unlab, x_unlab)
    unlab_loader = DataLoader(dataset=unlab_dataset, batch_size=parameters['batch_size'], shuffle=True)


    for epoch in range(parameters['iterations']):
        predictor.train()  # 切换到训练模式
        total_loss_epoch, total_supervised_loss_epoch, total_unsupervised_loss_epoch = 0, 0, 0

        train_iter = cycle(train_loader)
        for i, (xu_batch_ori, _) in enumerate(unlab_loader):
            x_batch, y_batch = next(train_iter)  # 获取有标签数据批次

            optimizer.zero_grad()  # 梯度清零

            with torch.no_grad():
                x_batch = encoder(x_batch)  # 编码有标签数据

            unsupervised_loss = 0
            # 初始化无标签数据批次列表
            for idx in range(K):
                m_batch = mask_generator(p_m, xu_batch_ori)
                _, xu_batch_temp = pretext_generator(m_batch, xu_batch_ori)

                with torch.no_grad():
                    xu_batch_temp = encoder(xu_batch_temp)  # 编码无标签数据

                # 计算无监督损失
                yv_hat_logit, yv_hat = predictor(xu_batch_temp)
                unsupervised_loss += unsupervised_loss_fn(yv_hat_logit)

            unsupervised_loss /= K

            # 计算有监督损失
            y_hat_logit, y_hat = predictor(x_batch)
            supervised_loss = supervised_loss_fn(y_hat_logit, y_batch)


            total_loss = supervised_loss + beta * unsupervised_loss
            total_loss.backward()  # 反向传播
            optimizer.step()  # 参数更新

            # 累积损失
            total_loss_epoch += total_loss.item()
            total_supervised_loss_epoch += supervised_loss.item()
            total_unsupervised_loss_epoch += unsupervised_loss.item()

        # 计算平均损失
        avg_supervised_loss = total_supervised_loss_epoch / len(unlab_loader)
        avg_unsupervised_loss = total_unsupervised_loss_epoch / len(unlab_loader)


        test_accuracy = compute_accuracy(predictor, encoder, x_test, y_test)

        print(
            f'Epoch {epoch + 1}, Avg Supervised Loss: {avg_supervised_loss:.4f}, Avg Unsupervised Loss: {avg_unsupervised_loss:.4f}, '
            f'Test Accuracy: {test_accuracy:.4f}%')


    # predictor.load_state_dict(torch.load(class_file_name))
    predictor.eval()  # 切换到评估模式

    # 在PyTorch中，进行预测时通常需要禁用梯度计算
    with torch.no_grad():
        y_test_hat_logit, y_test_hat = predictor(x_test)

    # 如果需要，转换y_test_hat为NumPy数组
    y_test_hat = np.argmax(y_test_hat.numpy(), axis=1)

    return y_test_hat


def compute_accuracy(predictor, encoder, x_test, y_test):
    predictor.eval()  # 切换到评估模式
    with torch.no_grad():  # 禁用梯度计算
        # x_test_encoded = encoder(x_test)  # 对测试集进行编码
        y_test_pred_logit, y_test_pred = predictor(x_test)  # 对测试集进行预测
        _, y_test_pred = torch.max(y_test_pred, dim=1)

        correct = (y_test_pred == y_test).sum().item()  # 计算正确预测的数量
        total = y_test.size(0)  # 测试集的总数
        accuracy = 100 * correct / total  # 计算准确率
    return accuracy


