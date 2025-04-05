import random

import numpy as np
import warnings

import torch
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")

from data_loader import load_mnist_data
from supervised_models_mlp import MLP, train_mlp_pytorch, predict_mlp_pytorch
from deae_self import DADE_Self
from deae_semi import train_model
from deae_utils import perf_metric

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)  # Set random seed for CPU
    np.random.seed(seed)  # Set random seed for NumPy module
    random.seed(seed)  # Set random seed for Python built-in random module
    torch.backends.cudnn.deterministic = True  # Ensure the convolution algorithm returned each time is deterministic
    torch.backends.cudnn.benchmark = False  # If the network input data dimension or type doesn't change much, setting True may increase running efficiency

model_sets = []

# Hyper-parameters
p_m = 0.3
alpha = 2
K = 3
beta = 0.1
# beta = 0
label_data_rate = 0.1
# Metric
metric = 'acc'

# Define output
results = np.zeros([len(model_sets) + 2])
x_train, y_train, x_unlab, x_test, y_test = load_mnist_data(label_data_rate)
file_name = './save_model/encoder_model'
_, dim = x_unlab.shape

# First, create a new encoder instance with the same structure as the saved encoder
encoder = DADE_Self(dim, alpha).encoder  # Assume dim and alpha are already defined

# MLP
input_dim = x_train.shape[1]
hidden_dim = 64  # Example hidden layer dimension
unique_labels = np.unique(y_train)
output_dim = len(unique_labels)
activation_fn = 'relu'

seed = 18

set_seed(seed)
model = MLP(input_dim, hidden_dim, output_dim, activation_fn)

mlp_parameters = {
    'batch_size': 128,
    'epochs': 100,
    'lr': 0.01,
}

# Load the saved state_dict
encoder.load_state_dict(torch.load(file_name))
encoder.eval()  # Switch to evaluation mode

with torch.no_grad():
    x_train_hat = encoder(x_train)
    x_test_hat = encoder(x_test)

train_mlp_pytorch(x_train_hat, y_train, model, mlp_parameters)
y_test_hat = predict_mlp_pytorch(x_test_hat, model)

# train_mlp_pytorch(x_train, y_train, model, mlp_parameters)
# y_test_hat = predict_mlp_pytorch(x_test, model)

results[0] = perf_metric(metric, y_test, y_test_hat)

print('DADE-Self Performance: ' + str(results[0]))

# Train DADE-Semi
dade_semi_parameters = dict()
dade_semi_parameters['hidden_dim'] = 64
dade_semi_parameters['batch_size'] = 128
# dade_semi_parameters['iterations'] = 149 # beta = 0.1
dade_semi_parameters['iterations'] = 109 # beta = 0
dade_semi_parameters['lr'] = 0.002

# for seed in range(20, 50):
#     print('seed ' + str(seed))

# set_seed(14)
y_test_hat = train_model(encoder, x_train, y_train, x_unlab, x_test, y_test,
                       dade_semi_parameters, p_m, K, beta)

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