import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import os
import matplotlib.pyplot as plt

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

def save_confusion_matrix(y_true, y_pred, output_dir, labels=None, normalize='true'):
  """Compute and save confusion matrix as CSV and PNG.

  Args:
    - y_true: array-like of true labels
    - y_pred: array-like of predicted labels
    - output_dir: directory to save outputs
    - labels: list of label names or ids (optional)
    - normalize: None, 'true', 'pred', or 'all' (sklearn normalization)
  """
  os.makedirs(output_dir, exist_ok=True)
  if labels is None:
    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))

  cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)

  # Save CSV
  csv_path = os.path.join(output_dir, 'confusion_matrix.csv')
  np.savetxt(csv_path, cm, delimiter=',', fmt='%.4f')

  # Save PNG
  fig, ax = plt.subplots(figsize=(6, 5))
  im = ax.imshow(cm, cmap='Blues')
  ax.set_xlabel('Predicted label')
  ax.set_ylabel('True label')
  ax.set_xticks(range(len(labels)))
  ax.set_yticks(range(len(labels)))
  ax.set_xticklabels(labels)
  ax.set_yticklabels(labels)
  plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
  for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
      ax.text(j, i, f'{cm[i, j]:.2f}', ha='center', va='center', color='black', fontsize=8)
  fig.colorbar(im, ax=ax)
  fig.tight_layout()
  png_path = os.path.join(output_dir, 'confusion_matrix.png')
  fig.savefig(png_path, dpi=200)
  plt.close(fig)

  return cm