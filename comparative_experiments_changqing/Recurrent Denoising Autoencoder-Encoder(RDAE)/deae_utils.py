import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

def mask_generator(p_m, x):
    # Generate a mask matrix using binomial distribution
    mask = np.random.binomial(1, p_m, x.shape)
    return mask

def pretext_generator(m, x):
    # Get the number of samples and dimensions
    no, dim = x.shape
    # Initialize a new matrix with the same shape as x
    x_bar = np.zeros([no, dim])
    # Shuffle each column independently
    for i in range(dim):
        idx = np.random.permutation(no)
        x_bar[:, i] = x[idx, i]

    # Corrupt samples
    x_tilde = x * (1 - m) + x_bar * m
    # Define new mask matrix
    m_new = 1 * (x != x_tilde)

    # Convert numpy arrays to torch tensors
    x_tilde_tensor = torch.tensor(x_tilde, dtype=torch.float32)
    m_new_tensor = torch.tensor(m_new, dtype=torch.float32)

    return m_new_tensor, x_tilde_tensor

#%% 
def perf_metric(metric, y_test, y_test_hat):
    if metric == 'acc':
        # Get the predicted labels by taking the index of the maximum value in each row
        predicted_labels = np.argmax(y_test_hat, axis=1)
        result = accuracy_score(y_test, predicted_labels)
    elif metric == 'auc':
        result = roc_auc_score(y_test, y_test_hat[:, 1])

    return result

#%% 
def convert_matrix_to_vector(matrix):
    """Convert a two-dimensional matrix into a one-dimensional vector

    Args:
        - matrix: a two-dimensional matrix

    Returns:
        - vector: a one-dimensional vector
    """
    # Get the number of samples and dimensions
    no, dim = matrix.shape
    # Initialize a vector with the same length as the number of samples
    vector = np.zeros([no,])

    # Convert matrix to vector
    for i in range(dim):
        idx = np.where(matrix[:, i] == 1)
        vector[idx] = i

    return vector

#%% 
def convert_vector_to_matrix(vector):
    """Convert a one-dimensional vector into a two-dimensional matrix

    Args:
        - vector: a one-dimensional vector

    Returns:
        - matrix: a two-dimensional matrix
    """
    # Get the number of samples
    no = len(vector)
    # Get the number of unique values in the vector
    dim = len(np.unique(vector))
    # Initialize a matrix with the same number of rows as the vector and columns as the number of unique values
    matrix = np.zeros([no, dim])

    # Convert vector to matrix
    for i in range(dim):
        idx = np.where(vector == i)
        matrix[idx, i] = 1

    return matrix