import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.semi_supervised import LabelPropagation

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score


def load_data(label_data_rate):
    seed = 15
    np.random.seed(seed)

    df = pd.read_csv('daqing.csv', encoding='gbk')  # Or use the correct encoding you know
    max_min = preprocessing.StandardScaler()

    # Split the training set and test set
    df_train = df[~df['Well_Name'].str.contains('Le')]
    df_test = df[df['Well_Name'].str.contains('Le')]

    # Select specific columns as features
    x_train = df_train.iloc[:, 3:16].values
    x_train = max_min.fit_transform(x_train)
    y_train = df_train['LITH'].values

    x_test = df_test.iloc[:, 3:16].values
    x_test = max_min.transform(x_test)  # Use the same scaler object to transform test data
    y_test = df_test['LITH'].values

    y_train -= 1  # Subtract 1 from labels to start encoding from 0
    y_test -= 1

    idx = np.random.permutation(x_train.shape[0])

    label_idx = idx[:int(len(idx) * label_data_rate)]
    unlab_idx = idx[int(len(idx) * label_data_rate):]

    x_label = x_train[label_idx, :]
    y_label = y_train[label_idx]

    x_unlab = x_train[unlab_idx, :]

    return x_label, y_label, x_unlab, x_test, y_test

# Assume we use a portion of the data as labeled data
x_label, y_label, x_unlab, x_test, y_test = load_data(0.1)

# Initialize the label propagation model
label_prop_model = LabelPropagation(max_iter=1000, kernel='rbf', gamma=5)

# Combine labeled and unlabeled data
x_total = np.vstack((x_label, x_unlab))
y_total = np.concatenate((y_label, -np.ones(x_unlab.shape[0])))

# Train the model
label_prop_model.fit(x_total, y_total)

# Make predictions
y_pred = label_prop_model.predict(x_test)

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
