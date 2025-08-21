import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

# Add parent directory to Python path for imports
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from evaluation_utils import save_confusion_matrix, print_classification_metrics

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Load and preprocess data
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = '/Users/zhouzihan/PycharmProjects/code-DEAE/data/daqing1.csv'

# Read data
df = pd.read_csv(data_path, encoding='utf-8-sig')

# Split data based on Well_Name
df_train = df[~df['Well_Name'].str.contains('Le')]
df_test = df[df['Well_Name'].str.contains('Le')]

# Prepare features and labels
x_train = torch.tensor(df_train.iloc[:, 3:16].values, dtype=torch.float32)
y_train = torch.tensor(df_train['LITH'].values - 1, dtype=torch.long)

x_test = torch.tensor(df_test.iloc[:, 3:16].values, dtype=torch.float32)
y_test = torch.tensor(df_test['LITH'].values - 1, dtype=torch.long)

# Load edge indices
edges_dir = os.path.join(current_dir, 'edges')
train_edges = np.loadtxt(os.path.join(edges_dir, 'daqing_train_edges.txt'), dtype=int)
test_edges = np.loadtxt(os.path.join(edges_dir, 'daqing_test_edges.txt'), dtype=int)

# Convert to PyTorch tensors
edge_index_train = torch.tensor(train_edges.T, dtype=torch.long)
edge_index_test = torch.tensor(test_edges.T, dtype=torch.long)

# Create Data objects
train_data = Data(x=x_train, edge_index=edge_index_train, y=y_train)
test_data = Data(x=x_test, edge_index=edge_index_test, y=y_test)

# Define GAT model
class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        x = self.lin(x)
        return F.log_softmax(x, dim=1)

# Initialize model
model = GAT(in_channels=13, hidden_channels=64, out_channels=6)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-6)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                      factor=0.5, patience=20,
                                                      verbose=True)

# Training loop
best_acc = 0.0
patience_counter = 0
max_patience = 50

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(train_data.x, train_data.edge_index)
    loss = F.nll_loss(out, train_data.y)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(test_data.x, test_data.edge_index).max(1)[1]
        acc_test = pred.eq(test_data.y).sum().item() / len(test_data.y)
        print(f'Epoch {epoch+1}, Test Accuracy: {acc_test:.4f}')

        scheduler.step(acc_test)
        if acc_test > best_acc:
            best_acc = acc_test
            patience_counter = 0
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

# Final evaluation
model.eval()
with torch.no_grad():
    pred = model(test_data.x, test_data.edge_index).max(1)[1]
    pred_np = pred.numpy()
    test_y_np = test_data.y.numpy()

# Print metrics and save confusion matrix
metrics = print_classification_metrics(test_y_np, pred_np, 'GAT')

# Save confusion matrix
output_dir = os.path.join(os.path.dirname(__file__), 'results')
save_confusion_matrix(test_y_np, pred_np, output_dir, 'gat')