import os
import shutil
import pandas as pd
import time
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Clear the NN_training_log folder
log_dir = 'NN_training_log'
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
os.makedirs(log_dir)

def nnArch(io=[9,1], hl=[12]):
    class NeuralNetwork(nn.Module):
        def __init__(self, hl):
            super(NeuralNetwork, self).__init__()
            self.layers = nn.ModuleList()
            input_size = io[0]
            
            # Create hidden layers
            for hidden_size in hl:
                self.layers.append(nn.Linear(input_size, hidden_size))
                self.layers.append(nn.ReLU())
                input_size = hidden_size
            
            # Create output layer
            self.layers.append(nn.Linear(input_size, io[1]))
            self.layers.append(nn.Sigmoid())
        
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
    
    # Create the neural network instance
    model = NeuralNetwork(hl)
    return model

def objective(trial):
    # Sample hyperparameters
    hidden_layers = trial.suggest_int('hidden_layers', 1, 4)
    hidden_units = [trial.suggest_int(f'n_units_l{i}', 4, 128) for i in range(hidden_layers)]
    learning_rate = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    
    # Create the model
    model = nnArch(io=[9, 1], hl=hidden_units).to(device)
    
    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize lists to store losses and validation accuracies
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # Training loop
    model.train()
    for epoch in range(10):  # Use more epochs for actual training
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
                
                # Calculate accuracy
                predicted = (outputs > 0.5).float()
                correct_val += (predicted == batch_y).sum().item()
                total_val += batch_y.size(0)
        
        epoch_val_loss /= len(val_loader)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(correct_val / total_val)
    
    # Save losses and validation accuracies to CSV file
    trial_number = trial.number
    df_metrics = pd.DataFrame({
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    })
    df_metrics.to_csv(f'NN_training_log/trial{trial_number}.csv', index=False)
    
    return val_losses[-1]

def logging_callback(study, trial):
    elapsed_time = time.time() - start_time
    trial_times.append(elapsed_time)
    avg_time_per_trial = sum(trial_times) / len(trial_times)
    trials_left = n_trials - trial.number - 1
    estimated_time_left = avg_time_per_trial * trials_left
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Estimated time left: {estimated_time_left:.2f} seconds")

def run_optimization():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, callbacks=[logging_callback])

    total_time = time.time() - start_time
    print(f"Total optimization time: {total_time:.2f} seconds")

    print('Best trial:')
    trial = study.best_trial
    print(f'  Value: {trial.value}')
    print('  Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')
    print(f'Best trial number: {trial.number}')

# Main execution logic
# Check if CUDA is available
print(f"Is CUDA supported by this system?  {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load the dataset
df = pd.read_csv('deleteme/dataset3D_50K_training_sampled.csv')

# Extract features (columns 1-9) and labels (column 10)
X = df.iloc[:, :9].values
y = df.iloc[:, 9].values

# Preprocess labels: if the value is non-zero, set it to 1.0
y = (y != 0).astype(float)

# Split the dataset into training and validation subsets (80:20 ratio)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # Ensure y is of shape (N, 1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)  # Ensure y is of shape (N, 1)

# Create DataLoaders
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=32, shuffle=False)

# Run the optimization
n_trials = 5
start_time = time.time()
trial_times = []

run_optimization()