import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from deae_utils import convert_matrix_to_vector, convert_vector_to_matrix

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation_fn='relu'):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        if activation_fn == 'relu':
            self.activation = F.relu
        elif activation_fn == 'tanh':
            self.activation = torch.tanh
        else:
            raise ValueError(f"Unknown activation function: {activation_fn}")

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

def train_mlp_pytorch(x_train, y_train, model, parameters):
    """Train MLP model using PyTorch."""
    batch_size = parameters['batch_size']
    epochs = parameters['epochs']
    lr = parameters['lr']

    # Convert data to PyTorch tensors
    x_train = torch.FloatTensor(x_train)
    y_train = torch.LongTensor(y_train)

    # Create data loader
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def predict_mlp_pytorch(x_test, model):
    """Make predictions using trained MLP model."""
    model.eval()
    with torch.no_grad():
        x_test = torch.FloatTensor(x_test)
        outputs = model(x_test)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.numpy()