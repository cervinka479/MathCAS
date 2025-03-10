import os
import shutil
import pandas as pd
import time
from datetime import timedelta
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.manual_seed(0)

def nnArch(io=[10,2], hl=[12], task='class'):    
    class NeuralNetwork(nn.Module):
        def __init__(self, hl):
            super(NeuralNetwork, self).__init__()
            self.layers = nn.ModuleList()
            input_size = io[0]
            
            # Create hidden layers
            for hidden_size in hl:
                self.layers.append(nn.Linear(input_size, hidden_size))
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(0.5))  # Add dropout for regularization
                input_size = hidden_size
            
            # Create output layer
            self.layers.append(nn.Linear(input_size, io[1]))
            if task == 'class':
                self.layers.append(nn.Sigmoid())
        
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
    
    # Create the neural network instance
    model = NeuralNetwork(hl)
    return model

def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, task='class'):
    # Initialize lists to store losses and validation accuracies
    train_losses = []
    val_losses = []
    val_accuracies = [] if task == 'class' else None
    
    # Variable to store the best validation loss
    best_val_loss = float('inf')
    patience = 10  # Number of epochs to wait for improvement
    patience_counter = 0
    
    # Training loop
    model.train()
    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)
    
        # Validation
        model.eval()
        epoch_val_loss = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                epoch_val_loss += loss.item()
                
                if task == 'class':
                    # Calculate accuracy
                    predicted = (outputs > 0.5).float()
                    correct_val += (predicted == batch_y).sum().item()
                    total_val += batch_y.size(0)
        
        epoch_val_loss /= len(val_loader)
        val_losses.append(epoch_val_loss)
        if task == 'class':
            val_accuracies.append(correct_val / total_val)

        # Save the model if it has the best validation loss so far
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), f'residual_best_model.pth')
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

        # Step the scheduler
        scheduler.step(epoch_val_loss)

        # Calculate elapsed time and estimated time left
        elapsed_time = time.time() - start_time
        epoch_time = time.time() - epoch_start_time
        estimated_time_left = epoch_time * (num_epochs - epoch - 1)
        
        # Format time in hours:minutes:seconds
        elapsed_time_str = str(timedelta(seconds=int(elapsed_time)))
        estimated_time_left_str = str(timedelta(seconds=int(estimated_time_left)))
        
        # Print epoch information
        if task == 'class':
            print(f"Epoch: {epoch + 1}/{num_epochs}, Val loss: {epoch_val_loss:.4f}, Accuracy: {correct_val / total_val:.4f}, Elapsed time: {elapsed_time_str}, Estimated time left: {estimated_time_left_str}")
        else:
            print(f"Epoch: {epoch + 1}/{num_epochs}, Val loss: {epoch_val_loss:.4f}, Elapsed time: {elapsed_time_str}, Estimated time left: {estimated_time_left_str}")
        sys.stdout.flush()

    # Print a newline after the final epoch information
    print()

    # Save losses and validation accuracies to CSV file
    df_metrics = pd.DataFrame({
        'train_losses': train_losses,
        'val_losses': val_losses,
    })
    if task == 'class':
        df_metrics['val_accuracies'] = val_accuracies
    df_metrics.to_csv(f'residual_metrics.csv', index=False)

# Main execution logic
# Check if CUDA is available
print(f"Is CUDA supported by this system?  {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Set the task
task = 'regression'  # Change this to 'class' for classification

# Load the dataset
df = pd.read_csv('deleteme\dataset_compressible_flow_5M_training_nstep180.csv')

# Extract features (columns 1-9 and 12) and labels (columns 10 and 11)
X = df.iloc[:, list(range(9)) + [11]].values
y = df.iloc[:, 9:11].values

# Preprocess labels: if the value is non-zero, set it to 1.0
if task == 'class':
    y = (y != 0).astype(float)

# Split the dataset into training and validation subsets (80:20 ratio)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# Create DataLoaders
batch_size = 168  # Set your batch size here
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)

# Set your hyperparameters here
hidden_layers = [460,460,460,460,460,460]
learning_rate = 8.33288590333646e-05
num_epochs = 500

# Create the model
model = nnArch(io=[10, 2], hl=hidden_layers, task=task).to(device)

# Define loss and optimizer
if task == 'class':
    criterion = nn.BCELoss()
else:
    criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

# Train and evaluate the model
train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, task=task)