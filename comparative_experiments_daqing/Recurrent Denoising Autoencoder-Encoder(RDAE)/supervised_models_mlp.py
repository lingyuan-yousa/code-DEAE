import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from deae_utils import convert_matrix_to_vector, convert_vector_to_matrix


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation_fn):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)

        if activation_fn == 'relu':
            self.activation = nn.ReLU()
        elif activation_fn == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError("Unsupported activation function")

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.layer3(x)  # No activation, assuming using CrossEntropyLoss which includes Softmax
        return x


def train_mlp_pytorch(x_train, y_train, model, parameters):
    if len(y_train.shape) > 1:
        y_train = convert_matrix_to_vector(y_train)

    # Create a data loader
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=parameters['batch_size'], shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=parameters['lr'])

    # Train the model
    model.train()
    for epoch in range(parameters['epochs']):
        correct = 0  # Number of correct predictions
        total = 0  # Total number of samples
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)  # Get the prediction result with the maximum probability
            total += labels.size(0)  # Update the total number of samples
            correct += (predicted == labels).sum().item()

        # epoch_accuracy = 100 * correct / total
        # print(f'Epoch [{epoch + 1}/{parameters["epochs"]}], Accuracy: {epoch_accuracy:.2f}%')


def predict_mlp_pytorch(x_test, model):
    model.eval()
    with torch.no_grad():
        y_test_hat = model(x_test)
        y_test_hat = torch.softmax(y_test_hat, dim=1)
    return y_test_hat.numpy()
