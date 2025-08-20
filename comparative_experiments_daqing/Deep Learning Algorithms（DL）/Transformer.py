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

class TransformerPredictor(nn.Module):
    def __init__(self, input_size, num_classes, d_model=64, nhead=8, 
                 num_layers=2, dim_feedforward=256, dropout=0.1):
        super(TransformerPredictor, self).__init__()
        
        # Linear layer to project input to d_model dimensions
        self.input_proj = nn.Linear(1, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(d_model * input_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, 1, input_size)
        
        # Reshape input for feature-wise projection
        x = x.transpose(1, 2)  # (batch_size, input_size, 1)
        
        # Project each feature to d_model dimensions
        x = self.input_proj(x)  # (batch_size, input_size, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)
        
        # Flatten and pass through final layers
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

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
    model = TransformerPredictor(
        input_size=input_size,
        num_classes=num_classes,
        d_model=64,
        nhead=8,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1
    )
    
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
                torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), 'results', 'best_transformer_model.pt'))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping triggered at epoch {epoch+1}')
                    break
            
            # Update learning rate scheduler
            scheduler.step(test_accuracy)
    
    # Load best model for final evaluation
    best_model_path = os.path.join(os.path.dirname(__file__), 'results', 'best_transformer_model.pt')
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
        metrics = print_classification_metrics(y_true, y_pred, 'Transformer')
        
        # Save confusion matrix
        output_dir = os.path.join(os.path.dirname(__file__), 'results')
        os.makedirs(output_dir, exist_ok=True)
        save_confusion_matrix(y_true, y_pred, output_dir, 'transformer', normalize='true')

if __name__ == '__main__':
    main()