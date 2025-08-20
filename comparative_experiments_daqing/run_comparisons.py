import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_model(script_path, model_name):
    """Run a model script and capture its output."""
    print(f"\nRunning {model_name}...")
    result = subprocess.run(['python', script_path], 
                          capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)

def collect_results(base_dir):
    """Collect all results and create comparison visualizations."""
    results = []
    
    # Collect metrics from all models
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.startswith('confusion_matrix_') and file.endswith('.csv'):
                model_name = file.replace('confusion_matrix_', '').replace('.csv', '')
                df = pd.read_csv(os.path.join(root, file))
                results.append({
                    'model': model_name,
                    'matrix': df
                })
    
    # Create comparison visualizations
    if results:
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.ravel()
        
        for i, result in enumerate(results):
            sns.heatmap(result['matrix'], 
                       ax=axes[i],
                       cmap='Blues',
                       annot=True,
                       fmt='.2f',
                       cbar=True)
            axes[i].set_title(f"{result['model'].upper()} Confusion Matrix")
            axes[i].set_xlabel('Predicted Lithology')
            axes[i].set_ylabel('True Lithology')
        
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, 'comparison_matrix.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

def main():
    # Define paths to model scripts
    base_dir = os.path.dirname(__file__)
    scripts = {
        'CatBoost': os.path.join(base_dir, 'Machine Learning Algorithms（ML）', 'CatBoost.py'),
        'GAT': os.path.join(base_dir, 'Graph Convolutional Network（GCN）', 'GAT.py'),
        'BiGRU': os.path.join(base_dir, 'Deep Learning Algorithms（DL）', 'RNN.py'),
        '1DCNN': os.path.join(base_dir, 'Deep Learning Algorithms（DL）', '1dcnn-mlp.py'),
    }
    
    # Run each model
    for model_name, script_path in scripts.items():
        if os.path.exists(script_path):
            run_model(script_path, model_name)
        else:
            print(f"Warning: Script not found - {script_path}")
    
    # Collect and visualize results
    collect_results(base_dir)

if __name__ == '__main__':
    main()
