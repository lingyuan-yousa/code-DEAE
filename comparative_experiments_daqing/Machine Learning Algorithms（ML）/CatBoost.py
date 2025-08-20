import os
import sys
from catboost import CatBoostClassifier, Pool
import numpy as np
from sklearn.model_selection import GridSearchCV

# Add parent directory to Python path for imports
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from data_loader import load_data
from evaluation_utils import save_confusion_matrix, print_classification_metrics

def train_with_grid_search(x_train, y_train, x_test, y_test):
    # Create parameter grid
    param_grid = {
        'learning_rate': [0.01, 0.03, 0.05],
        'depth': [4, 6, 8],
        'l2_leaf_reg': [1, 3, 5],
        'bagging_temperature': [0.5, 1.0],
        'random_strength': [0.5, 1.0],
    }
    
    # Base model
    base_model = CatBoostClassifier(
        iterations=1000,  # Will be used for initial grid search
        early_stopping_rounds=50,
        loss_function='MultiClass',
        eval_metric='Accuracy',
        verbose=100,
        task_type='CPU',
        bootstrap_type='Bayesian'  # Enable Bayesian bootstrap
    )
    
    # Create train and validation pools
    train_pool = Pool(x_train, y_train)
    test_pool = Pool(x_test, y_test)
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,  # Use all available cores
        verbose=2
    )
    
    print("Starting Grid Search...")
    grid_search.fit(x_train, y_train)
    
    print("\nBest parameters:", grid_search.best_params_)
    print("Best CV accuracy:", grid_search.best_score_)
    
    # Train final model with best parameters and more iterations
    best_params = grid_search.best_params_
    final_model = CatBoostClassifier(
        iterations=3000,  # Increased iterations for final model
        early_stopping_rounds=100,
        learning_rate=best_params['learning_rate'],
        depth=best_params['depth'],
        l2_leaf_reg=best_params['l2_leaf_reg'],
        bagging_temperature=best_params['bagging_temperature'],
        random_strength=best_params['random_strength'],
        loss_function='MultiClass',
        eval_metric='Accuracy',
        verbose=100,
        task_type='CPU',
        bootstrap_type='Bayesian',
        random_seed=42  # Set seed for reproducibility
    )
    
    print("\nTraining final model with best parameters...")
    final_model.fit(train_pool, eval_set=test_pool, use_best_model=True)
    
    return final_model

def main():
    # Load data
    x_label, y_label, x_unlab, x_test, y_test = load_data(0.3)
    
    # Train model with grid search
    model = train_with_grid_search(x_label, y_label, x_test, y_test)
    
    # Make predictions
    y_pred = model.predict(x_test)
    y_pred = y_pred.reshape(-1)  # Flatten predictions if needed
    
    # Print metrics and save confusion matrix
    metrics = print_classification_metrics(y_test, y_pred, 'CatBoost')
    
    # Save confusion matrix
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)
    save_confusion_matrix(y_test, y_pred, output_dir, 'catboost', normalize='true')
    
    # Save best model
    model_path = os.path.join(output_dir, 'best_catboost_model.cbm')
    model.save_model(model_path)
    print(f"\nBest model saved to: {model_path}")
    
    # Print feature importances
    feature_importances = model.get_feature_importance()
    feature_names = [f'feature_{i}' for i in range(len(feature_importances))]
    print("\nFeature Importances:")
    for name, importance in zip(feature_names, feature_importances):
        print(f"{name}: {importance:.4f}")

if __name__ == '__main__':
    main()