import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from comparative_experiments_daqing.data_loader import load_mnist_data
from torch_geometric.data import Data

x_label, y_label, x_unlab, x_test, y_test = load_mnist_data(0.3)


class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the output of the last time step
        return out


data = Data(x=x_label, y=y_label)
test_data = Data(x=x_test, y=y_test)

input_size = data.x.shape[1]
num_classes = len(np.unique(y_label))

# Initialize the LSTM model
model = LSTMPredictor(input_size, hidden_size=64, num_layers=2, num_classes=num_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 300

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    optimizer.zero_grad()

    outputs = model(data.x.unsqueeze(1))
    loss = criterion(outputs.squeeze(), data.y)
    loss.backward()
    optimizer.step()

    # Evaluate on the test set at the end of each epoch
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        test_outputs = model(test_data.x.unsqueeze(1))
        _, test_predicted = torch.max(test_outputs, 1)

        test_correct = (test_predicted == test_data.y).sum().item()
        test_accuracy = test_correct / len(test_data.y)
        print(f'Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {test_accuracy * 100:.2f}%')

model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    test_outputs = model(test_data.x.unsqueeze(1))
    _, pred = torch.max(test_outputs, 1)

    test_correct = (test_predicted == test_data.y).sum().item()
    test_accuracy = test_correct / len(test_data.y)
    print(f'Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {test_accuracy * 100:.2f}%')

y_pred = pred

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