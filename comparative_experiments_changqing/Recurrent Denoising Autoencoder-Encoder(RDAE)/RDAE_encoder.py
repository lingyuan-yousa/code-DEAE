import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import random

from comparative_experiments_changqing.DenoisingAuto_Encoder.deae_utils import perf_metric
from supervised_models_mlp import MLP, train_mlp_pytorch, predict_mlp_pytorch


class RecurrentDenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=32):
        super(RecurrentDenoisingAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.output_dim = input_dim

        # Encoder
        self.encoder = nn.LSTM(input_dim, input_dim, batch_first=True)

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x, noise_factor=0.5):
        # Add noise
        noisy_x = x + noise_factor * torch.randn_like(x)

        # Encode
        _, (hidden, _) = self.encoder(noisy_x.unsqueeze(1))

        # Decode
        decoded = self.decoder(hidden.squeeze(0))

        return decoded

    def set_seed(self, seed):
        """Set random seed for reproducibility"""
        torch.manual_seed(seed)  # Set random seed for CPU
        np.random.seed(seed)  # Set random seed for NumPy module
        random.seed(seed)  # Set random seed for Python built-in random module
        torch.backends.cudnn.deterministic = True  # Ensure the convolution algorithm returned each time is deterministic
        torch.backends.cudnn.benchmark = False  # If the network input data dimension or type doesn't change much, setting True may increase running efficiency

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

                # Calculate loss
                loss = criterion(output, xu_batch)  # Ensure feature_pred and x_tilde have the same dimensions

                # Backward propagation and optimization
                loss.backward()
                optimizer.step()

                # Accumulate loss
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
                x_train_hat1, (_, _) = self.encoder(x_train)
                x_test_hat1, (_, _) = self.encoder(x_test)

            train_mlp_pytorch(x_train_hat1, y_train, model, mlp_parameters)
            y_test_hat1 = predict_mlp_pytorch(x_test_hat1, model)

            acc = perf_metric('acc', y_test, y_test_hat1)

            print('RDAE Performance: ' + str(acc))

            # Print the loss after each epoch
            print(f'Epoch {epoch + 1}/{epochs}, Total Loss: {total_loss:.4f}')