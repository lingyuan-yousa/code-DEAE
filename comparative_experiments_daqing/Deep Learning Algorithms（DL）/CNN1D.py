import os
import sys
import numpy as np
import torch
import torch.nn as nn

# Add parent directory to Python path for imports
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from data_loader import load_mnist_data as load_data
from evaluation_utils import save_confusion_matrix, print_classification_metrics

class CNN1DPredictor(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNN1DPredictor, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Second convolutional layer
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Calculate the size of flattened features
        self.flat_features = 64 * (input_size // 4)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.flat_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Input shape: (batch_size, 1, input_size)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, self.flat_features)
        x = self.fc(x)
        return x

def main():
    # Load data
    x_label, y_label, x_unlab, x_test, y_test = load_data(0.3)
    
    # Convert data to tensors
    x_label = torch.FloatTensor(x_label).unsqueeze(1)  # Add channel dimension
    y_label = torch.LongTensor(y_label)
    x_test = torch.FloatTensor(x_test).unsqueeze(1)
    y_test = torch.LongTensor(y_test)
    
    # Model parameters
    input_size = x_label.shape[2]  # Number of features
    num_classes = len(np.unique(y_label))
    
    # Initialize the model
    model = CNN1DPredictor(input_size, num_classes)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training parameters
    num_epochs = 300
    batch_size = 32
    best_acc = 0.0
    patience = 50  # Early stopping patience
    patience_counter = 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                          factor=0.5, patience=20, 
                                                          verbose=True)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        
        # Simple batch training
        for i in range(0, len(x_label), batch_size):
            batch_x = x_label[i:i+batch_size]
            batch_y = y_label[i:i+batch_size]
            
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(x_test)
            _, predicted = torch.max(test_outputs.data, 1)
            test_accuracy = (predicted == y_test).sum().item() / len(y_test)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {test_accuracy:.4f}')
            
            # Early stopping check
            if test_accuracy > best_acc:
                best_acc = test_accuracy
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), 'results', 'best_cnn_model.pt'))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping triggered at epoch {epoch+1}')
                    break
            
            # Update learning rate scheduler
            scheduler.step(test_accuracy)
    
    # Load best model for final evaluation
    best_model_path = os.path.join(os.path.dirname(__file__), 'results', 'best_cnn_model.pt')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print("Loaded best model for evaluation")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(x_test)
        _, predicted = torch.max(test_outputs.data, 1)
        
        # Convert predictions to numpy for evaluation
        y_pred = predicted.numpy()
        y_true = y_test.numpy()
        
        # Print metrics and save confusion matrix
        metrics = print_classification_metrics(y_true, y_pred, '1D-CNN')
        
        # Save confusion matrix
        output_dir = os.path.join(os.path.dirname(__file__), 'results')
        os.makedirs(output_dir, exist_ok=True)
        save_confusion_matrix(y_true, y_pred, output_dir, '1dcnn', normalize='true')

if __name__ == '__main__':
    main()