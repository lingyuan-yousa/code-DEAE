import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import random

from VIME.vime_utils import perf_metric
from supervised_models_mlp import MLP, train_mlp_pytorch, predict_mlp_pytorch

class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=64):
        super(DenoisingAutoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.LeakyReLU(True),
            nn.Linear(encoding_dim, input_dim),
            nn.LeakyReLU(True)
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.LeakyReLU(True),
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x, noise_factor=0.5):
        # 添加噪声
        noisy_x = x + noise_factor * torch.randn_like(x)
        # 编码
        encoded = self.encoder(noisy_x)
        # 解码
        decoded = self.decoder(encoded)
        return decoded

    def set_seed(self, seed):
        """设置随机种子以确保可重复性"""
        torch.manual_seed(seed)  # 为CPU设置随机种子
        np.random.seed(seed)  # Numpy模块的随机种子
        random.seed(seed)  # Python内置的随机模块
        torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
        torch.backends.cudnn.benchmark = False  # 如果网络输入数据维度或类型上变化不大，设置True可能会增加运行效率

    def train_model(self, x_unlab, p_m, parameters, x_train, y_train, x_test, y_test, seed):

        # 创建TensorDataset和DataLoader
        dataset = TensorDataset(x_unlab, x_unlab)
        batch_size = parameters['batch_size']
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=0.002)
        criterion = nn.MSELoss()

        epochs = parameters['epochs']

        for epoch in range(epochs):
            total_loss = 0

            for xu_batch, _ in dataloader:  # xu_batch 直接从 DataLoader 中获取

                optimizer.zero_grad()

                # 前向传播
                output = self.forward(xu_batch)

                # 计算损失
                loss = criterion(output, xu_batch)  # 确保feature_pred和x_tilde尺寸一致

                # 反向传播和优化
                loss.backward()
                optimizer.step()

                # 累计损失
                total_loss += loss.item()

            # MLP
            input_dim = x_train.shape[1]
            hidden_dim = 64  # 示例隐藏层维度
            unique_labels = np.unique(y_train)
            output_dim = len(unique_labels)
            activation_fn = 'relu'

            self.set_seed(seed)
            model = MLP(input_dim, hidden_dim, output_dim, activation_fn)

            mlp_parameters = {
                'batch_size': 128,
                'epochs': 100,
                'lr': 0.002,
            }

            self.encoder.eval()  # 切换到评估模式

            with torch.no_grad():
                x_train_hat1 = self.encoder(x_train)
                x_test_hat1 = self.encoder(x_test)

            train_mlp_pytorch(x_train_hat1, y_train, model, mlp_parameters)
            y_test_hat1 = predict_mlp_pytorch(x_test_hat1, model)

            acc = perf_metric('acc', y_test, y_test_hat1)

            print('DAE Performance: ' + str(acc))

            # 每个epoch结束后打印损失
            print(
                f'Epoch {epoch + 1}/{epochs}, Total Loss: {total_loss:.4f}')
