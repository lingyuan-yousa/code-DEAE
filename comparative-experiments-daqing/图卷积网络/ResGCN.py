import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn import preprocessing
import random
import torch.nn as nn

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

seed = 4
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# 加载数据
data = pd.read_csv("E:/well/data/daqing/daqing.csv", encoding='GBK')

# 创建一个唯一的岩性名称到数字类别的映射字典
unique_liths = data['LITH'].unique()
lith_to_category = {lith: category for category, lith in enumerate(unique_liths)}

# 使用replace方法将LITH列中的中文标签替换为数字类别
data['LITH'] = data['LITH'].replace(lith_to_category)

# 选取包含关键字 '乐' 的行作为测试集的下标
test_idx = data[data['Well_Name'].str.contains('乐')].index

# 选取不包含关键字 '乐' 的行作为训练集的下标
train_idx = data[~data['Well_Name'].str.contains('乐')].index

# 从表格中提取特征和标签
x = data.iloc[train_idx, 3:16].values
max_min = preprocessing.StandardScaler()
x = max_min.fit_transform(x)
x = torch.tensor(x, dtype=torch.float)

test_x = data.iloc[test_idx, 3:16].values
test_x = max_min.transform(test_x)
test_x = torch.tensor(test_x, dtype=torch.float)

edges = [[i, i] for i in range(len(x))]
test_edges = [[i, i] for i in range(len(test_x))]

edges = torch.tensor(edges, dtype=torch.long).t().contiguous()
test_edges = torch.tensor(test_edges, dtype=torch.long).t().contiguous()

y = data.iloc[train_idx, -1].values
y = torch.tensor(y, dtype=torch.long)

test_y = data.iloc[test_idx, -1].values
test_y = torch.tensor(test_y, dtype=torch.long)

# 创建PyTorch Geometric的Data对象
graph_data = Data(x=x, edge_index=edges, y=y)
test_graph_data = Data(x=test_x, edge_index=test_edges, y=test_y)

# 划分有标签和无标签数据
num_nodes = graph_data.num_nodes
num_labeled = int(num_nodes * 0.1)
all_indices = list(range(num_nodes))
random.shuffle(all_indices)
labeled_indices = all_indices[:num_labeled]

hid = 64


# 定义GNN模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(graph_data.num_features, hid)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.conv2 = GCNConv(hid + graph_data.num_features, hid)
        self.conv3 = GCNConv(hid + graph_data.num_features, len(torch.unique(graph_data.y)))  # 使用y的唯一值数量作为输出类的数量

        self.conv2 = GCNConv(hid + graph_data.num_features, hid)
        self.conv3 = GCNConv(hid + graph_data.num_features, len(torch.unique(graph_data.y)))  # 使用y的唯一值数量作为输出类的数量


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        tx = x

        x = F.relu(self.conv1(x, edge_index))
        x = torch.cat([x, tx], dim=1)

        x = self.dropout2(x)
        x = F.relu(self.conv2(x, edge_index))
        x = torch.cat([x, tx], dim=1)

        x = self.dropout3(x)
        x = self.conv3(x, edge_index)
        return x


# 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

initial_lr = 0.0009
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=5e-6)

# 进行训练
for epoch in range(68): # 698 643
    model.train()
    optimizer.zero_grad()
    out = model(graph_data)
    loss = F.cross_entropy(out[labeled_indices], graph_data.y[labeled_indices])
    loss.backward()
    optimizer.step()

    model.eval()
    _, pred = model(graph_data).max(dim=1)
    correct = pred[labeled_indices].eq(graph_data.y[labeled_indices]).sum().item()
    accuracy_train = correct / len(labeled_indices)

    _, pred = model(test_graph_data).max(dim=1)
    correct = pred.eq(test_graph_data.y).sum().item()
    accuracy_test = correct / test_graph_data.num_nodes

    print('Seed: {:02d}'.format(seed),
          'Epoch: {:04d}'.format(epoch + 1),
          'acc_train: {:.4f}'.format(accuracy_train),
          'acc_val: {:.4f}'.format(accuracy_test))

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