import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from deae_utils import mask_generator, pretext_generator  # Ensure these functions are compatible with PyTorch

from itertools import cycle

# class PredictorModel(nn.Module):
#     def __init__(self, data_dim, hidden_dim, label_dim):
#         super(PredictorModel, self).__init__()
#         self.network = nn.Sequential(
#             nn.Linear(data_dim, hidden_dim),
#             # nn.Dropout(p=0.01),
#             nn.LeakyReLU(),
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             # nn.Dropout(p=0.01),
#             nn.LeakyReLU(),
#             nn.Linear(hidden_dim // 2, label_dim)
#         )
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         y_hat_logit = self.network(x)
#         y_hat = self.softmax(y_hat_logit)
#         return y_hat_logit, y_hat

class PredictorModel(nn.Module):
    def __init__(self, data_dim, hidden_dim, label_dim):
        super(PredictorModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),  # Assume data_dim is large enough
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * (data_dim // 4), hidden_dim),  # Need to adjust according to CNN output
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, label_dim),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.cnn(x)
        x = x.view(x.size(0), -1)  # Flatten
        y_hat_logit = self.fc(x)
        y_hat = self.softmax(y_hat_logit)
        return y_hat_logit, y_hat


def unsupervised_loss_fn(yv_hat_logit):
    # Calculate the variance along the specified dimension (here, across the batch for each feature)
    variance = torch.var(yv_hat_logit, dim=0, unbiased=False)
    # Calculate the mean of the variance as the unsupervised loss
    return torch.mean(variance)

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)  # Set random seed for CPU
    np.random.seed(seed)  # Set random seed for NumPy module
    random.seed(seed)  # Set random seed for Python built-in random module
    torch.backends.cudnn.deterministic = True  # Ensure the convolution algorithm returned each time is deterministic
    torch.backends.cudnn.benchmark = False  # If the network input data dimension or type doesn't change much, setting True may increase running efficiency


# def train_model(encoder, x_train, y_train, x_unlab, x_test, y_test, parameters, p_m, K, beta, seed):
def train_model(encoder, x_train, y_train, x_unlab, x_test, y_test, parameters, p_m, K, beta):
    hidden_dim = parameters['hidden_dim']

    # Basic parameters
    data_dim = len(x_train[0, :])
    label_dim = len(np.unique(y_train))

    # Training settings
    predictor = PredictorModel(data_dim, hidden_dim, label_dim)
    optimizer = optim.Adam(predictor.parameters(), lr=parameters['lr'], weight_decay=0, amsgrad=False)
    supervised_loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        x_test = encoder(x_test)

    # Data preparation
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=parameters['batch_size'], shuffle=True)

    unlab_dataset = TensorDataset(x_unlab, x_unlab)
    unlab_loader = DataLoader(dataset=unlab_dataset, batch_size=parameters['batch_size'], shuffle=True)


    for epoch in range(parameters['iterations']):
        predictor.train()  # Switch to training mode
        total_loss_epoch, total_supervised_loss_epoch, total_unsupervised_loss_epoch = 0, 0, 0

        train_iter = cycle(train_loader)
        for i, (xu_batch_ori, _) in enumerate(unlab_loader):
            x_batch, y_batch = next(train_iter)  # Get a batch of labeled data

            optimizer.zero_grad()  # Zero the gradients

            with torch.no_grad():
                x_batch = encoder(x_batch)  # Encode the labeled data

            unsupervised_loss = 0
            # Initialize a list for unlabeled data batches
            for idx in range(K):
                m_batch = mask_generator(p_m, xu_batch_ori)
                _, xu_batch_temp = pretext_generator(m_batch, xu_batch_ori)

                with torch.no_grad():
                    xu_batch_temp = encoder(xu_batch_temp)  # Encode the unlabeled data

                # Calculate the unsupervised loss
                yv_hat_logit, yv_hat = predictor(xu_batch_temp)
                unsupervised_loss += unsupervised_loss_fn(yv_hat_logit)

            unsupervised_loss /= K

            # Calculate the supervised loss
            y_hat_logit, y_hat = predictor(x_batch)
            supervised_loss = supervised_loss_fn(y_hat_logit, y_batch)


            total_loss = supervised_loss + beta * unsupervised_loss
            total_loss.backward()  # Backpropagation
            optimizer.step()  # Parameter update

            # Accumulate losses
            total_loss_epoch += total_loss.item()
            total_supervised_loss_epoch += supervised_loss.item()
            total_unsupervised_loss_epoch += unsupervised_loss.item()

        # Calculate average losses
        avg_supervised_loss = total_supervised_loss_epoch / len(unlab_loader)
        avg_unsupervised_loss = total_unsupervised_loss_epoch / len(unlab_loader)


        test_accuracy = compute_accuracy(predictor, encoder, x_test, y_test)

        print(
            f'Epoch {epoch + 1}, Avg Supervised Loss: {avg_supervised_loss:.4f}, Avg Unsupervised Loss: {avg_unsupervised_loss:.4f}, '
            f'Test Accuracy: {test_accuracy:.4f}%')


    # predictor.load_state_dict(torch.load(class_file_name))
    predictor.eval()  # Switch to evaluation mode

    # In PyTorch, gradient computation is usually disabled during prediction
    with torch.no_grad():
        y_test_hat_logit, y_test_hat = predictor(x_test)

    # Convert y_test_hat to a NumPy array if needed
    y_test_hat = np.argmax(y_test_hat.numpy(), axis=1)

    return y_test_hat


def compute_accuracy(predictor, encoder, x_test, y_test):
    predictor.eval()  # Switch to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        # x_test_encoded = encoder(x_test)  # Encode the test set
        y_test_pred_logit, y_test_pred = predictor(x_test)  # Make predictions on the test set
        _, y_test_pred = torch.max(y_test_pred, dim=1)

        correct = (y_test_pred == y_test).sum().item()  # Calculate the number of correct predictions
        total = y_test.size(0)  # Total number of samples in the test set
        accuracy = 100 * correct / total  # Calculate the accuracy
    return accuracy


