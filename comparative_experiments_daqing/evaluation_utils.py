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

def print_classification_metrics(y_true, y_pred, model_name):
    """Print classification metrics including accuracy, precision, recall, and F1 score."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}\n")
    print("Detailed Classification Report:")
    print(classification_report(y_true, y_pred))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }