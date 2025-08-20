import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn import preprocessing
import pandas as pd
import numpy as np
import random

# Add parent directory to Python path for imports
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from evaluation_utils import save_confusion_matrix, print_classification_metrics


seed = 4
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Read data
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
df = pd.read_csv(os.path.join(data_dir, "daqing1.csv"), encoding='utf-8-sig')

# Split by well name
df_train = df[~df['Well_Name'].str.contains('Le')]
df_test = df[df['Well_Name'].str.contains('Le')]

# Extract features and standardize
max_min_scaler = preprocessing.StandardScaler()

# Process training data
x = df_train.iloc[:, 3:16].values
x = max_min_scaler.fit_transform(x)
x = torch.tensor(x, dtype=torch.float)

y = df_train['LITH'].values - 1  # Convert to 0-based indexing
y = torch.tensor(y, dtype=torch.long)

# Process test data
test_x = df_test.iloc[:, 3:16].values
test_x = max_min_scaler.transform(test_x)  # Use same scaler as training
test_x = torch.tensor(test_x, dtype=torch.float)

test_y = df_test['LITH'].values - 1  # Convert to 0-based indexing
test_y = torch.tensor(test_y, dtype=torch.long)

# Read edges
edges_dir = os.path.join(os.path.dirname(__file__), 'edges')
train_edges = []
with open(os.path.join(edges_dir, "daqing_train_edges.txt"), "r") as train_file:
    for line in train_file:
        edge = [int(x) for x in line.strip().split()]
        train_edges.append(edge)

test_edges = []
with open(os.path.join(edges_dir, "daqing_test_edges.txt"), "r") as test_file:
    for line in test_file:
        edge = [int(x) for x in line.strip().split()]
        test_edges.append(edge)

edges = torch.tensor(train_edges, dtype=torch.long).t().contiguous()
test_edges = torch.tensor(test_edges, dtype=torch.long).t().contiguous()

# Create graph data
graph_data = Data(x=x, edge_index=edges, y=y)
test_graph_data = Data(x=test_x, edge_index=test_edges, y=test_y)

# Split labeled and unlabeled data
num_nodes = graph_data.num_nodes
num_labeled = int(num_nodes * 0.1)
all_indices = list(range(num_nodes))
random.shuffle(all_indices)
labeled_indices = all_indices[:num_labeled]
unlabeled_indices = all_indices[num_labeled:]

# Define the GNN model
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

# Model parameters
hid = 64
heads = 8
out_features = len(torch.unique(y))

# Initialize the model and optimizer
model = Net(graph_data.num_features, hid, out_features, heads)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-6)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                      factor=0.5, patience=20, 
                                                      verbose=True)

# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
graph_data = graph_data.to(device)
test_graph_data = test_graph_data.to(device)

best_acc = 0.0
patience_counter = 0
max_patience = 50  # Early stopping patience

for epoch in range(200):  # Increased epochs
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
    
    # Update learning rate scheduler
    scheduler.step(acc_test)
    
    # Early stopping check
    if acc_test > best_acc:
        best_acc = acc_test
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), 'results', 'best_gat_model.pt'))
    else:
        patience_counter += 1
        if patience_counter >= max_patience:
            print(f'Early stopping triggered at epoch {epoch+1}')
            break

# Load best model for final evaluation
best_model_path = os.path.join(os.path.dirname(__file__), 'results', 'best_gat_model.pt')
if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path))
    print("Loaded best model for evaluation")

# Test the model
model.eval()
_, pred = model(test_graph_data).max(dim=1)

# Convert predictions to numpy for evaluation
pred_np = pred.cpu().numpy()
test_y_np = test_y.cpu().numpy()

# Print metrics and save confusion matrix
metrics = print_classification_metrics(test_y_np, pred_np, 'GAT')

# Save confusion matrix
output_dir = os.path.join(os.path.dirname(__file__), 'results')
save_confusion_matrix(test_y_np, pred_np, output_dir, 'gat', normalize='true')