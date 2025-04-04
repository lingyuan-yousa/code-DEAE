import lightgbm as lgb
import numpy as np

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

from data_loader import load_data

x_label, y_label, x_unlab, x_test, y_test = load_data(0.3)

# Set LightGBM parameters
params = {
    'objective': 'multiclass',
    'num_class': len(np.unique(y_label)),  # Number of your classes
    'learning_rate': 0.01,
    'num_leaves': 35,  # Appropriate number of leaf nodes
    'max_depth': -1,  # Appropriate maximum depth
    'min_child_samples': 20,  # Appropriate minimum number of child samples
    'subsample': 0.8,  # Appropriate subsampling ratio
    'colsample_bytree': 0.8,  # Appropriate column sampling ratio
    'reg_alpha': 1,  # Appropriate regularization parameter
    'reg_lambda': 1,  # Appropriate regularization parameter
}

# Initialize the LightGBM model
model = lgb.LGBMClassifier(**params, n_estimators=100)

# Train the LightGBM model
model.fit(x_label, y_label)

# Make predictions on the test set
y_pred = model.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
# Calculate precision
precision = precision_score(y_test, y_pred, average='weighted')
# Calculate recall
recall = recall_score(y_test, y_pred, average='weighted')
# Calculate F1 score
f1_score_value = f1_score(y_test, y_pred, average='weighted')

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1_score_value:.4f}')

report = classification_report(y_test, y_pred)
print(report)
