import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn import preprocessing
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score


seed = 4
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


df = pd.read_csv('changqing.csv', encoding='utf-8')  # 或使用你知道的正确编码

train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# 提取特征和标签
x = train_data.iloc[:, 2:13].values
max_min_scaler = preprocessing.StandardScaler()
x = max_min_scaler.fit_transform(x)
x = torch.tensor(x, dtype=torch.float)

test_x = test_data.iloc[:, 2:13].values
test_x = max_min_scaler.transform(test_x)
test_x = torch.tensor(test_x, dtype=torch.float)

y = train_data.iloc[:, -1].values - 1
y = torch.tensor(y, dtype=torch.long)

test_y = test_data.iloc[:, -1].values - 1
test_y = torch.tensor(test_y, dtype=torch.long)

# 读取边
path = "E:/well/code/daqing建边/"
train_edges = []
with open(path + "daqing_train_edges_depth_feature.txt", "r") as train_file:
    for line in train_file:
        edge = [int(x) for x in line.strip().split()]
        train_edges.append(edge)

test_edges = []
with open(path + "daqing_test_edges_depth_feature.txt", "r") as test_file:
    for line in test_file:
        edge = [int(x) for x in line.strip().split()]
        test_edges.append(edge)

edges = torch.tensor(train_edges, dtype=torch.long).t().contiguous()
test_edges = torch.tensor(test_edges, dtype=torch.long).t().contiguous()

# 创建图数据
graph_data = Data(x=x, edge_index=edges, y=y)
test_graph_data = Data(x=test_x, edge_index=test_edges, y=test_y)

# 划分有标签和无标签数据
num_nodes = graph_data.num_nodes
num_labeled = int(num_nodes * 0.3)
all_indices = list(range(num_nodes))
random.shuffle(all_indices)
labeled_indices = all_indices[:num_labeled]
unlabeled_indices = all_indices[num_labeled:]

# 定义GNN模型
class Net(torch.nn.Module):
    def __init__(self, in_features, hid, out_features, heads):
        super(Net, self).__init__()
        self.conv1 = GATConv(in_features, hid, heads=heads, dropout=0)
        self.conv2 = GATConv(hid * heads, hid, heads=heads, dropout=0.05)
        self.conv3 = GATConv(hid * heads, out_features, heads=1, dropout=0.19)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))

        x = F.relu(self.conv2(x, edge_index))

        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

# 模型参数
hid = 64
heads = 8
out_features = len(torch.unique(y))

# 初始化模型和优化器
model = Net(graph_data.num_features, hid, out_features, heads)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-6)

# 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
graph_data = graph_data.to(device)
test_graph_data = test_graph_data.to(device)

for epoch in range(271):
    model.train()
    optimizer.zero_grad()

    out = model(graph_data)
    loss = F.cross_entropy(out[labeled_indices], graph_data.y[labeled_indices])
    loss.backward()
    optimizer.step()

    model.eval()
    _, pred = model(graph_data).max(dim=1)
    correct = pred[labeled_indices].eq(graph_data.y[labeled_indices]).sum().item()
    acc_train = correct / len(labeled_indices)

    _, pred = model(test_graph_data).max(dim=1)
    correct = pred.eq(test_graph_data.y).sum().item()
    acc_test = correct / test_graph_data.num_nodes

    print(f'Epoch: {epoch+1}, Train Acc: {acc_train:.4f}, Test Acc: {acc_test:.4f}')


# 测试模型
model.eval()
_, pred = model(test_graph_data).max(dim=1)

# 计算准确率
accuracy = accuracy_score(test_y, pred)

# 计算精确率
precision = precision_score(test_y, pred, average='weighted')  # 'macro' 考虑了所有类别的均衡

# 计算召回率
recall = recall_score(test_y, pred, average='weighted')

# 计算F1分数
f1 = f1_score(test_y, pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")