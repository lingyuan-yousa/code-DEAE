from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

from data_loader import load_data

x_label, y_label, x_unlab, x_test, y_test = load_data(0.3)

n_neighbors = 6

# Initialize the KNN model
knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)

# Train the KNN model
knn_model.fit(x_label, y_label)

# Make predictions on the test set
y_pred = knn_model.predict(x_test)

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
