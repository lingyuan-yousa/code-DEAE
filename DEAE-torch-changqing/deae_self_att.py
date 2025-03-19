import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.functional import multi_head_attention_forward

import random

from supervised_models_mlp import MLP, train_mlp_pytorch, predict_mlp_pytorch
from deae_utils import mask_generator, pretext_generator
from deae_utils import perf_metric


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0, bias=True):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, bias=bias)

    def forward(self, query):
        # 为了使用 MultiheadAttention，需要query的形状是 (L, N, E)
        # L 是序列长度，N 是批处理大小，E 是特征数量（嵌入维度）
        # 如果输入query的形状是 (N, E)，需要使用unsqueeze(0)来添加序列长度维度，例如 (1, N, E)
        query = query.unsqueeze(0)

        # 多头注意力模型期望键（key），值（value）和查询（query）具有相同的维度
        attn_output, _ = self.multihead_attn(query, query, query)

        # 将输出的形状从 (L, N, E) 改回 (N, E)
        attn_output = attn_output.squeeze(0)
        return attn_output

class VIME_Self(nn.Module):
    def __init__(self, dim, alpha, num_heads=4, hid1=64, hid2=64):
        super(VIME_Self, self).__init__()

        self.attention = MultiHeadAttention(embed_dim=hid1, num_heads=num_heads)

        self.encoder = nn.Sequential(
            nn.Linear(dim, dim),  # 最后一个线性层将维度降回原始维度
            nn.LeakyReLU()
        )

        # Mask estimator
        self.mask_estimator = nn.Sequential(
            nn.Linear(dim, hid1),  # 第一个隐藏层，dim为输入维度，128为该层的输出维度
            nn.LeakyReLU(),  # 第一个隐藏层的ReLU激活函数
            self.attention,  # 注意力层
            nn.Linear(hid1, hid2),  # 第二个隐藏层，128为输入维度，64为该层的输出维度
            nn.LeakyReLU(),  # 第二个隐藏层的ReLU激活函数
            nn.Linear(hid2, dim),  # 输出层，64为输入维度，dim为输出维度，保持与原模型输出尺寸一致
            nn.Sigmoid()  # 输出层的Sigmoid激活函数，用于二分类概率预测
        )


        self.feature_estimator = nn.Sequential(
            nn.Linear(dim, hid1),  # 第一个隐藏层，dim为输入维度，128为该层的输出维度
            nn.LeakyReLU(),  # 第一个隐藏层的ReLU激活函数
            self.attention,  # 注意力层
            nn.Linear(hid1, hid2),  # 第二个隐藏层，128为输入维度，64为该层的输出维度
            nn.LeakyReLU(),  # 第二个隐藏层的ReLU激活函数
            nn.Linear(hid2, dim),  # 输出层，64为输入维度，dim为输出维度，保持与原模型输出尺寸一致
            nn.Sigmoid()  # 输出层的Sigmoid激活函数，用于二分类概率预测
        )

        self.alpha = alpha

    def forward(self, x):
        h = self.encoder(x)

        mask_output = self.mask_estimator(h)
        feature_output = self.feature_estimator(h)

        return mask_output, feature_output

    def train_model(self, x_unlab, p_m, parameters, x_train, y_train, x_test, y_test, seed):

        # 创建TensorDataset和DataLoader
        dataset = TensorDataset(x_unlab, x_unlab)
        batch_size = parameters['batch_size']
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=0.001)
        criterion_mask = nn.BCELoss()
        criterion_feature = nn.MSELoss()

        epochs = parameters['epochs']

        for epoch in range(epochs):
            total_loss = 0
            total_mask_loss = 0
            total_feature_loss = 0
            for xu_batch, _ in dataloader:  # xu_batch 直接从 DataLoader 中获取

                optimizer.zero_grad()

                # 根据自己的情况调整mask_generator和pretext_generator的使用
                m_unlab = mask_generator(p_m, xu_batch)
                m_label, x_tilde = pretext_generator(m_unlab, xu_batch)

                # 前向传播
                mask_pred, feature_pred = self.forward(x_tilde)

                # 计算损失
                loss_mask = criterion_mask(mask_pred, m_label)
                loss_feature = criterion_feature(feature_pred, xu_batch)  # 确保feature_pred和x_tilde尺寸一致
                loss = loss_mask + self.alpha * loss_feature

                # 反向传播和优化
                loss.backward()
                optimizer.step()

                # 累计损失
                total_loss += loss.item()
                total_mask_loss += loss_mask.item()
                total_feature_loss += loss_feature.item()


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
                'lr': 0.01,
            }

            self.encoder.eval()  # 切换到评估模式

            with torch.no_grad():
                x_train_hat1 = self.encoder(x_train)
                x_test_hat1 = self.encoder(x_test)

            train_mlp_pytorch(x_train_hat1, y_train, model, mlp_parameters)
            y_test_hat1 = predict_mlp_pytorch(x_test_hat1, model)

            acc = perf_metric('acc', y_test, y_test_hat1)

            print('VIME-Self Performance: ' + str(acc))

            # 每个epoch结束后打印损失
            print(
                f'Epoch {epoch + 1}/{epochs}, Total Loss: {total_loss:.4f}, Mask Loss: {total_mask_loss:.4f}, Feature Loss: {total_feature_loss:.4f}')

    def set_seed(self, seed):
        """设置随机种子以确保可重复性"""
        torch.manual_seed(seed)  # 为CPU设置随机种子
        np.random.seed(seed)  # Numpy模块的随机种子
        random.seed(seed)  # Python内置的随机模块
        torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
        torch.backends.cudnn.benchmark = False  # 如果网络输入数据维度或类型上变化不大，设置True可能会增加运行效率