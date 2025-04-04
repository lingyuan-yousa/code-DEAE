import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import random

from supervised_models_mlp import MLP, train_mlp_pytorch, predict_mlp_pytorch
from deae_utils import mask_generator, pretext_generator
from deae_utils import perf_metric

class DADE_Self(nn.Module):
    def __init__(self, dim, alpha):
        super(DADE_Self, self).__init__()
        # Encoder
        # self.encoder = nn.Sequential(
        #     nn.Linear(dim, 128),  # Expand to twice the dimension to provide more model capacity
        #     nn.LeakyReLU(),
        #     nn.Linear(128, 64),  # Add an additional hidden layer
        #     nn.LeakyReLU(),
        #     nn.Linear(64, dim),  # The last linear layer reduces the dimension back to the original dimension
        #     nn.LeakyReLU()
        # )

        self.encoder = nn.Sequential(
            nn.Linear(dim, dim),  # The last linear layer reduces the dimension back to the original dimension
            nn.LeakyReLU()
        )

        # Mask estimator
        self.mask_estimator = nn.Sequential(
            nn.Linear(dim, 128),  # The first hidden layer, dim is the input dimension, 128 is the output dimension of this layer
            nn.LeakyReLU(),  # ReLU activation function for the first hidden layer
            nn.Linear(128, 64),  # The second hidden layer, 128 is the input dimension, 64 is the output dimension of this layer
            nn.LeakyReLU(),  # ReLU activation function for the second hidden layer
            nn.Linear(64, dim),  # Output layer, 64 is the input dimension, dim is the output dimension, keeping the same size as the original model output
            nn.Sigmoid()  # Sigmoid activation function for the output layer, used for binary classification probability prediction
        )

        # Feature estimator
        # self.feature_estimator = nn.Sequential(
        #     nn.Linear(dim, dim),
        #     nn.Sigmoid()
        # )

        self.feature_estimator = nn.Sequential(
            nn.Linear(dim, 128),  # The first hidden layer, dim is the input dimension, 128 is the output dimension of this layer
            nn.LeakyReLU(),  # ReLU activation function for the first hidden layer
            nn.Linear(128, 64),  # The second hidden layer, 128 is the input dimension, 64 is the output dimension of this layer
            nn.LeakyReLU(),  # ReLU activation function for the second hidden layer
            nn.Linear(64, dim),  # Output layer, 64 is the input dimension, dim is the output dimension, keeping the same size as the original model output
            nn.Sigmoid()  # Sigmoid activation function for the output layer, used for binary classification probability prediction
        )

        self.alpha = alpha

    def forward(self, x):
        h = self.encoder(x)

        mask_output = self.mask_estimator(h)
        feature_output = self.feature_estimator(h)

        return mask_output, feature_output

    def train_model(self, x_unlab, p_m, parameters, x_train, y_train, x_test, y_test, seed):

        # Create TensorDataset and DataLoader
        dataset = TensorDataset(x_unlab, x_unlab)
        batch_size = parameters['batch_size']
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=0.01)
        criterion_mask = nn.BCELoss()
        criterion_feature = nn.MSELoss()

        epochs = parameters['epochs']

        for epoch in range(epochs):
            total_loss = 0
            total_mask_loss = 0
            total_feature_loss = 0
            for xu_batch, _ in dataloader:  # xu_batch is directly obtained from the DataLoader

                optimizer.zero_grad()

                # Adjust the use of mask_generator and pretext_generator according to your own situation
                m_unlab = mask_generator(p_m, xu_batch)
                m_label, x_tilde = pretext_generator(m_unlab, xu_batch)

                # Forward propagation
                mask_pred, feature_pred = self.forward(x_tilde)

                # Calculate loss
                loss_mask = criterion_mask(mask_pred, m_label)
                loss_feature = criterion_feature(feature_pred, xu_batch)  # Ensure that feature_pred and x_tilde have the same size
                loss = loss_mask + self.alpha * loss_feature

                # Backpropagation and optimization
                loss.backward()
                optimizer.step()

                # Accumulate loss
                total_loss += loss.item()
                total_mask_loss += loss_mask.item()
                total_feature_loss += loss_feature.item()


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
                'lr': 0.01,
            }

            self.encoder.eval()  # Switch to evaluation mode

            with torch.no_grad():
                x_train_hat1 = self.encoder(x_train)
                x_test_hat1 = self.encoder(x_test)

            train_mlp_pytorch(x_train_hat1, y_train, model, mlp_parameters)
            y_test_hat1 = predict_mlp_pytorch(x_test_hat1, model)

            acc = perf_metric('acc', y_test, y_test_hat1)

            print('DADE-Self Performance: ' + str(acc))

            # Print loss after each epoch
            print(
                f'Epoch {epoch + 1}/{epochs}, Total Loss: {total_loss:.4f}, Mask Loss: {total_mask_loss:.4f}, Feature Loss: {total_feature_loss:.4f}')

    def set_seed(self, seed):
        """Set random seed for reproducibility"""
        torch.manual_seed(seed)  # Set random seed for CPU
        np.random.seed(seed)  # Set random seed for NumPy module
        random.seed(seed)  # Set random seed for Python built-in random module
        torch.backends.cudnn.deterministic = True  # Ensure the convolution algorithm returned each time is deterministic
        torch.backends.cudnn.benchmark = False  # If the network input data dimension or type doesn't change much, setting True may increase running efficiency        torch.backends.cudnn.benchmark = False  # 如果网络输入数据维度或类型上变化不大，设置True可能会增加运行效率