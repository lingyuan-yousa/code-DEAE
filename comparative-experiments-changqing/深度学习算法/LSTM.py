import random

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from data_loader import load_mnist_data
from torch_geometric.data import Data

x_label, y_label, x_unlab, x_test, y_test = load_mnist_data(0.3)


class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out


data = Data(x=x_label, y=y_label)
test_data = Data(x=x_test, y=y_test)

input_size = data.x.shape[1]
num_classes = len(np.unique(y_label))

# 初始化LSTM模型
model = LSTMPredictor(input_size, hidden_size=64, num_layers=2, num_classes=num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 420

for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    optimizer.zero_grad()

    outputs = model(data.x.unsqueeze(1))
    loss = criterion(outputs.squeeze(), data.y)
    loss.backward()
    optimizer.step()

    # 在每个epoch结束时进行测试集的评估
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        test_outputs = model(test_data.x.unsqueeze(1))
        _, test_predicted = torch.max(test_outputs, 1)

        test_correct = (test_predicted == test_data.y).sum().item()
        test_accuracy = test_correct / len(test_data.y)
        print(f'Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {test_accuracy * 100:.2f}%')

model.eval()  # 设置模型为评估模式
with torch.no_grad():
    test_outputs = model(test_data.x.unsqueeze(1))
    _, pred = torch.max(test_outputs, 1)

    test_correct = (test_predicted == test_data.y).sum().item()
    test_accuracy = test_correct / len(test_data.y)
    print(f'Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {test_accuracy * 100:.2f}%')

y_pred = pred

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
# 计算精确率
precision = precision_score(y_test, y_pred, average='weighted')
# 计算召回率
recall = recall_score(y_test, y_pred, average='weighted')
# 计算 F1 分数
f1_score_value = f1_score(y_test, y_pred, average='weighted')

print(f'准确率：{accuracy:.4f}')
print(f'精确率：{precision:.4f}')
print(f'召回率：{recall:.4f}')
print(f'F1 分数：{f1_score_value:.4f}')

report = classification_report(y_test, y_pred)
print(report)