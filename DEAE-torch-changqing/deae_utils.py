import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

def mask_generator (p_m, x):
  mask = np.random.binomial(1, p_m, x.shape)
  return mask

def pretext_generator (m, x):
  no, dim = x.shape

  x_bar = np.zeros([no, dim])
  for i in range(dim):
    idx = np.random.permutation(no)
    x_bar[:, i] = x[idx, i]

  # Corrupt samples
  x_tilde = x * (1-m) + x_bar * m
  # Define new mask matrix
  m_new = 1 * (x != x_tilde)

  x_tilde_tensor = torch.tensor(x_tilde, dtype=torch.float32)
  m_new_tensor = torch.tensor(m_new, dtype=torch.float32)  # Or choose an appropriate data type according to the actual situation

  return m_new_tensor, x_tilde_tensor

#%%
def perf_metric (metric, y_test, y_test_hat):
  if metric == 'acc':
    predicted_labels = np.argmax(y_test_hat, axis=1)
    result = accuracy_score(y_test, predicted_labels)
  elif metric == 'auc':
    result = roc_auc_score(y_test, y_test_hat[:, 1])

  return result

#%%
def convert_matrix_to_vector(matrix):
  """Convert two dimensional matrix into one dimensional vector

  Args:
    - matrix: two dimensional matrix

  Returns:
    - vector: one dimensional vector
  """
  # Parameters
  no, dim = matrix.shape
  # Define output
  vector = np.zeros([no,])

  # Convert matrix to vector
  for i in range(dim):
    idx = np.where(matrix[:, i] == 1)
    vector[idx] = i

  return vector

#%%
def convert_vector_to_matrix(vector):
  """Convert one dimensional vector into two dimensional matrix

  Args:
    - vector: one dimensional vector

  Returns:
    - matrix: two dimensional matrix
  """
  # Parameters
  no = len(vector)
  dim = len(np.unique(vector))
  # Define output
  matrix = np.zeros([no,dim])

  # Convert vector to matrix
  for i in range(dim):
    idx = np.where(vector == i)
    matrix[idx, i] = 1

  return matrix

