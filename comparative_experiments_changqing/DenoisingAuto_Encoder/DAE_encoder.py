import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import random

from comparative_experiments_changqing.DenoisingAuto_Encoder.deae_utils import perf_metric
from supervised_models_mlp import MLP, train_mlp_pytorch, predict_mlp_pytorch

class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=64):
        super(DenoisingAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.LeakyReLU(True),
            nn.Linear(encoding_dim, input_dim),
            nn.LeakyReLU(True)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.LeakyReLU(True),
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x, noise_factor=0.5):
        # Add noise
        noisy_x = x + noise_factor * torch.randn_like(x)
        # Encode
        encoded = self.encoder(noisy_x)
        # Decode
        decoded = self.decoder(encoded)
        return decoded

    def set_seed(self, seed):
        """Set the random seed to ensure reproducibility"""
        torch.manual_seed(seed)  # Set the random seed for the CPU
        np.random.seed(seed)  # Random seed for the Numpy module
        random.seed(seed)  # Random seed for the built-in Python random module
        torch.backends.cudnn.deterministic = True  # Ensure that the convolution algorithm returned each time is deterministic
        torch.backends.cudnn.benchmark = False  # If the network input data dimension or type does not change much, setting True may increase the running efficiency

    def train_model(self, x_unlab, p_m, parameters, x_train, y_train, x_test, y_test, seed):

        # Create TensorDataset and DataLoader
        dataset = TensorDataset(x_unlab, x_unlab)
        batch_size = parameters['batch_size']
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=0.002)
        criterion = nn.MSELoss()

        epochs = parameters['epochs']

        for epoch in range(epochs):
            total_loss = 0

            for xu_batch, _ in dataloader:  # xu_batch is directly obtained from the DataLoader

                optimizer.zero_grad()

                # Forward propagation
                output = self.forward(xu_batch)

                # Calculate the loss
                loss = criterion(output, xu_batch)  # Ensure that the dimensions of feature_pred and x_tilde are consistent

                # Backpropagation and optimization
                loss.backward()
                optimizer.step()

                # Accumulate the loss
                total_loss += loss.item()

            # MLP
            input_dim = x_train.shape[1]
            hidden_dim = 64  # Example hidden layer dimension
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

            self.encoder.eval()  # Switch to evaluation mode

            with torch.no_grad():
                x_train_hat1 = self.encoder(x_train)
                x_test_hat1 = self.encoder(x_test)

            train_mlp_pytorch(x_train_hat1, y_train, model, mlp_parameters)
            y_test_hat1 = predict_mlp_pytorch(x_test_hat1, model)

            acc = perf_metric('acc', y_test, y_test_hat1)

            print('DAE Performance: ' + str(acc))

            # Print the loss after each epoch
            print(
                f'Epoch {epoch + 1}/{epochs}, Total Loss: {total_loss:.4f}')
