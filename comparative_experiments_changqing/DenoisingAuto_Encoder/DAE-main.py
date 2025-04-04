import random

import numpy as np
import os
import warnings

import torch
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score


warnings.filterwarnings("ignore")

from comparative_experiments_changqing.data_loader import load_mnist_data
from supervised_models_mlp import MLP, train_mlp_pytorch, predict_mlp_pytorch

from DAE_encoder import DenoisingAutoencoder

from deae_utils import perf_metric

def set_seed(seed):
    """Set the random seed to ensure reproducibility"""
    torch.manual_seed(seed)  # Set the random seed for the CPU
    np.random.seed(seed)  # Random seed for the Numpy module
    random.seed(seed)  # Random seed for the built-in Python random module
    torch.backends.cudnn.deterministic = True  # Ensure that the convolution algorithm returned each time is deterministic
    torch.backends.cudnn.benchmark = False  # If the network input data dimension or type does not change much, setting True may increase the running efficiency

model_sets = []

# Hyper-parameters
p_m = 0.3
alpha = 2
K = 3
beta = 0.1
label_data_rate = 0.3

# Metric
metric = 'acc'

# Define output
results = np.zeros([len(model_sets) + 2])

x_train, y_train, x_unlab, x_test, y_test = load_mnist_data(label_data_rate)

# Train VIME-Self
vime_self_parameters = dict()
vime_self_parameters['batch_size'] = 32
vime_self_parameters['epochs'] = 100

file_name = './save_model/encoder_model'

# for seed in range(1, 30):
#     print('seed ' + str(seed))

seed = 10 #41

set_seed(seed)
_, dim = x_unlab.shape
DAE_model = DenoisingAutoencoder(dim)
DAE_model.train_model(x_unlab, p_m, vime_self_parameters, x_train, y_train, x_test, y_test, seed)

# Save encoder
if not os.path.exists('save_model'):
    os.makedirs('save_model')

torch.save(DAE_model.encoder.state_dict(), file_name)

# First, create a new encoder instance with the same structure as the saved encoder
encoder = DenoisingAutoencoder(dim).encoder  # Assume dim and alpha are already defined

# MLP
input_dim = x_train.shape[1]
hidden_dim = 64  # Example hidden layer dimension
unique_labels = np.unique(y_train)
output_dim = len(unique_labels)
activation_fn = 'relu'

set_seed(seed)
model = MLP(input_dim, hidden_dim, output_dim, activation_fn)

mlp_parameters = {
    'batch_size': 128,
    'epochs': 100,
    'lr': 0.002,
}

# Load the saved state_dict
encoder.load_state_dict(torch.load(file_name))
encoder.eval()  # Switch to evaluation mode

with torch.no_grad():
    x_train_hat = encoder(x_train)
    x_test_hat = encoder(x_test)


train_mlp_pytorch(x_train_hat, y_train, model, mlp_parameters)
y_test_hat = predict_mlp_pytorch(x_test_hat, model)

y_test_hat = np.argmax(y_test_hat, axis=1)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_test_hat)
# Calculate precision
precision = precision_score(y_test, y_test_hat, average='weighted')
# Calculate recall
recall = recall_score(y_test, y_test_hat, average='weighted')
# Calculate F1 score
f1_score_value = f1_score(y_test, y_test_hat, average='weighted')

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1_score_value:.4f}')

report = classification_report(y_test, y_test_hat)
print(report)



