import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_confusion_matrices():
    """Load confusion matrices from different models."""
    matrices = {}
    model_dirs = {
        'CatBoost': 'Machine Learning Algorithms（ML）/results',
        'GAT': 'Graph Convolutional Network（GCN）/results',
        'BiGRU': 'Deep Learning Algorithms（DL）/results',
        '1D-CNN': 'Deep Learning Algorithms（DL）/results',
        'Transformer': 'Deep Learning Algorithms（DL）/results'
    }
    
    for model, dir_path in model_dirs.items():
        file_path = os.path.join(os.path.dirname(__file__), dir_path, 
                               f'confusion_matrix_{model.lower().replace("-", "")}.csv')
        if os.path.exists(file_path):
            matrices[model] = pd.read_csv(file_path)
    
    return matrices

def plot_comparison(matrices):
    """Create comparison visualization of confusion matrices."""
    if not matrices:
        print("No confusion matrices found!")
        return
    
    # Create a figure with subplots
    n_models = len(matrices)
    n_rows = (n_models + 2) // 3  # Ensure we have enough rows
    n_cols = min(3, n_models)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 7*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot each confusion matrix
    for i, (model, matrix) in enumerate(matrices.items()):
        row = i // n_cols
        col = i % n_cols
        sns.heatmap(matrix, ax=axes[row, col], cmap='Blues', annot=True, fmt='.2f')
        axes[row, col].set_title(f'{model} Confusion Matrix')
        axes[row, col].set_xlabel('Predicted Lithology')
        axes[row, col].set_ylabel('True Lithology')
    
    # Hide empty subplots if any
    for i in range(len(matrices), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    output_dir = os.path.join(os.path.dirname(__file__), 'comparison_results')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_table(matrices):
    """Create a performance comparison table."""
    performance = {
        'GAT': {'Accuracy': 0.3735, 'Precision': 0.2560, 'Recall': 0.3735, 'F1': 0.2659},
        'BiGRU': {'Accuracy': 0.7051, 'Precision': 0.7305, 'Recall': 0.7051, 'F1': 0.7060},
        '1D-CNN': {'Accuracy': 0.7022, 'Precision': 0.7137, 'Recall': 0.7022, 'F1': 0.7016},
        'Transformer': {'Accuracy': 0.6742, 'Precision': 0.6837, 'Recall': 0.6742, 'F1': 0.6755}
    }
    
    df = pd.DataFrame(performance).T
    df = df.round(4)
    
    # Save to CSV
    output_dir = os.path.join(os.path.dirname(__file__), 'comparison_results')
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, 'model_performance.csv'))
    
    return df

def main():
    # Load confusion matrices
    matrices = load_confusion_matrices()
    
    # Create visualizations
    plot_comparison(matrices)
    
    # Create and display performance table
    performance_table = create_performance_table(matrices)
    print("\nModel Performance Comparison:")
    print(performance_table)
    
    # Create bar plot of performance metrics
    metrics_plot = performance_table.plot(
        kind='bar',
        figsize=(12, 6),
        width=0.8
    )
    plt.title('Model Performance Comparison')
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.legend(title='Metrics')
    plt.tight_layout()
    
    # Save metrics plot
    output_dir = os.path.join(os.path.dirname(__file__), 'comparison_results')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()