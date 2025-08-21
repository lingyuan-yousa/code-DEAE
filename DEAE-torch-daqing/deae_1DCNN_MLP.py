import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from deae_utils import mask_generator, pretext_generator

class CNN1D_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CNN1D_MLP, self).__init__()
        
        # 1D CNN layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the size after CNN layers
        cnn_output_size = 64 * (input_dim // 4)  # Divided by 4 because of two pooling layers
        
        # MLP layers
        self.fc1 = nn.Linear(cnn_output_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Ensure input is float32
        x = x.float()
        
        # Reshape input for 1D CNN (batch_size, channels, length)
        x = x.unsqueeze(1)
        
        # CNN layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # MLP layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

def train_model(encoder, x_train, y_train, x_unlab, x_test, y_test, parameters, p_m, K, beta):
    """Train the CNN1D-MLP model."""
    hidden_dim = parameters['hidden_dim']
    batch_size = parameters['batch_size']
    iterations = parameters['iterations']
    lr = parameters['lr']

    # Get dimensions
    _, input_dim = x_train.shape
    unique_labels = np.unique(y_train)
    output_dim = len(unique_labels)

    # Convert data to PyTorch tensors
    x_train = torch.FloatTensor(x_train)
    y_train = torch.LongTensor(y_train)
    x_unlab = torch.FloatTensor(x_unlab)
    x_test = torch.FloatTensor(x_test)

    # Create data loaders
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    unlab_loader = DataLoader(x_unlab, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = CNN1D_MLP(input_dim, hidden_dim, output_dim)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for iteration in range(iterations):
        model.train()
        
        # Get a batch of labeled data
        try:
            batch_x, batch_y = next(iter(train_loader))
        except StopIteration:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            batch_x, batch_y = next(iter(train_loader))
        
        # Get a batch of unlabeled data
        try:
            batch_u = next(iter(unlab_loader))
        except StopIteration:
            unlab_loader = DataLoader(x_unlab, batch_size=batch_size, shuffle=True)
            batch_u = next(iter(unlab_loader))

        # Generate masked data
        m_batch = mask_generator(batch_u.numpy(), p_m)
        x_tilde = pretext_generator(batch_u.numpy(), m_batch)
        x_tilde = torch.FloatTensor(x_tilde)

        # Forward pass
        outputs = model(batch_x)
        loss_s = criterion(outputs, batch_y)

        # Unsupervised loss
        loss_u = 0
        for k in range(K):
            m_batch = mask_generator(batch_u.numpy(), p_m)
            x_tilde = pretext_generator(batch_u.numpy(), m_batch)
            x_tilde = torch.FloatTensor(x_tilde)
            pred_u = model(x_tilde)
            with torch.no_grad():
                pred = model(batch_u)
            loss_u += criterion(pred_u, pred.max(1)[1])
        loss_u /= K

        # Total loss
        loss = loss_s + beta * loss_u

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(x_test)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.numpy()