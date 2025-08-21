import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, 
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

def convert_matrix_to_vector(matrix):
    """Convert a matrix to a vector."""
    return matrix.reshape(-1)

def convert_vector_to_matrix(vector, shape):
    """Convert a vector back to a matrix with the specified shape."""
    return vector.reshape(shape)

def mask_generator(x, p_m):
    """Generate a mask for the input data."""
    n, d = x.shape
    mask = np.random.choice([0, 1], size=(n, d), p=[p_m, 1 - p_m])
    return mask

def pretext_generator(x, mask):
    """Generate pretext task data."""
    n, d = x.shape
    noise = np.random.normal(0, 1, (n, d))
    x_tilde = x * mask + noise * (1 - mask)
    return x_tilde

def perf_metric(metric, y_true, y_pred):
    """Calculate performance metric."""
    if metric == 'acc':
        return accuracy_score(y_true, y_pred)
    elif metric == 'precision':
        return precision_score(y_true, y_pred, average='weighted')
    elif metric == 'recall':
        return recall_score(y_true, y_pred, average='weighted')
    elif metric == 'f1':
        return f1_score(y_true, y_pred, average='weighted')
    else:
        raise ValueError(f"Unknown metric: {metric}")

def save_confusion_matrix(y_true, y_pred, output_dir, model_name):
    """Save confusion matrix as CSV and PNG with proper lithology labels.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        output_dir: Directory to save outputs
        model_name: Name of the model (used in filename)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define lithology names
    lithology_names = {
        0: "Silty Mudstone",          # SS
        1: "Heterogeneous Sandstone", # HS
        2: "Dark Mudstone",           # DS
        3: "Homogeneous Sandstone",   # HgS
        4: "Black Shale",             # BS
        5: "Tuff"                     # Tuff
    }
    
    labels = sorted(list(np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))))
    label_names = [lithology_names[label] for label in labels]
    
    # Compute confusion matrix (raw counts)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Save raw counts to CSV
    csv_path = os.path.join(output_dir, f'confusion_matrix_{model_name}.csv')
    
    # Save with header and row names for better readability
    with open(csv_path, 'w') as f:
        # Write header
        f.write('True Label,' + ','.join(label_names) + '\n')
        # Write rows with actual counts
        for i, row in enumerate(cm):
            f.write(f'{label_names[i]},' + ','.join(map(str, row)) + '\n')
    
    # Save raw counts to Excel-friendly CSV
    excel_path = os.path.join(output_dir, f'confusion_matrix_{model_name}_excel.csv')
    np.savetxt(excel_path, cm, delimiter=',', fmt='%d')
    
    # Create figure with larger size and higher DPI
    plt.figure(figsize=(12, 10), dpi=300)
    
    # Create heatmap with scientific color scheme
    im = plt.imshow(cm, cmap='YlOrRd')  # Using YlOrRd colormap for better scientific presentation
    
    # Add title
    plt.title(f'Confusion Matrix - {model_name}', pad=20, fontsize=14)
    
    # Add axis labels
    plt.xlabel('Predicted Label', fontsize=12, labelpad=10)
    plt.ylabel('True Label', fontsize=12, labelpad=10)
    
    # Add ticks
    plt.xticks(range(len(labels)), label_names, rotation=45, ha='right', fontsize=10)
    plt.yticks(range(len(labels)), label_names, fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Number of Samples',
                   rotation=270, labelpad=25, fontsize=10)
    
    # Add text annotations with actual counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            plt.text(j, i, str(cm[i, j]),
                    ha="center", va="center", color=text_color,
                    fontsize=10, fontweight='bold')
    
    # Adjust layout and save
    plt.tight_layout()
    png_path = os.path.join(output_dir, f'confusion_matrix_{model_name}.png')
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return cm