import os
import shutil
import pandas as pd
import time
from datetime import timedelta
import logging
import sys
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.manual_seed(0)

# Clear the NN_residual_training_log folder
log_dir = 'NN_correction_training_log'
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
os.makedirs(log_dir)

# Configure the logging
log_dir = 'NN_correction_training_log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file = os.path.join(log_dir, 'output.log')

# Create a custom logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Create handlers
console_handler = logging.StreamHandler(sys.stdout)
file_handler = logging.FileHandler(log_file)

# Set logging level for handlers
console_handler.setLevel(logging.DEBUG)
file_handler.setLevel(logging.DEBUG)

# Create formatters and add them to handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Redirect stdout and stderr to the logging system
class StreamToLogger(object):
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

# Redirect stdout and stderr
sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
sys.stderr = StreamToLogger(logging.getLogger('STDERR'), logging.ERROR)

# Configure Optuna logger to use the same handlers
optuna.logging.get_logger("optuna").addHandler(file_handler)


def nnArch(io=[12,3], hl=[12], dropout_prob=0.5, task='class'):    
    class NeuralNetwork(nn.Module):
        def __init__(self, hl):
            super(NeuralNetwork, self).__init__()
            self.layers = nn.ModuleList()
            input_size = io[0]
            
            # Create hidden layers with dropout
            for hidden_size in hl:
                self.layers.append(nn.Linear(input_size, hidden_size))
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(dropout_prob))  # Add dropout for regularization
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

def objective(trial, task='class'):
    # Sample hyperparameters
    hidden_layers = trial.suggest_int('hidden_layers', 1, 8)
    hidden_units = trial.suggest_int('n_units_l', 128, 512, log=True)
    hidden_units_list = [int(hidden_units)] * hidden_layers
    learning_rate = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_int('batch_size', 16, 256, log=True)

    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)
    
    # Create the model
    model = nnArch(io=[12, 3], hl=hidden_units_list, dropout_prob=0.5, task=task).to(device)
    
    # Define loss and optimizer
    if task == 'class':
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
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
    for epoch in range(num_epochs):  # Use sampled number of epochs for training
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
            torch.save(model.state_dict(), f'NN_correction_training_log/trial{trial.number}_best_model.pth')
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
            print(f"Trial: {trial.number + 1}/{n_trials}, Epoch: {epoch + 1}/{num_epochs}, Val loss: {epoch_val_loss:.4f}, Accuracy: {correct_val / total_val:.4f}, Elapsed time: {elapsed_time_str}, Estimated time left: {estimated_time_left_str}")
        else:
            print(f"Trial: {trial.number + 1}/{n_trials}, Epoch: {epoch + 1}/{num_epochs}, Val loss: {epoch_val_loss:.4f}, Elapsed time: {elapsed_time_str}, Estimated time left: {estimated_time_left_str}")
        sys.stdout.flush()

    # Print a newline after the final epoch information
    print()

    # Save losses and validation accuracies to CSV file
    trial_number = trial.number
    df_metrics = pd.DataFrame({
        'train_losses': train_losses,
        'val_losses': val_losses,
    })
    if task == 'class':
        df_metrics['val_accuracies'] = val_accuracies
    df_metrics.to_csv(f'NN_correction_training_log/trial{trial_number}.csv', index=False)
    
    return val_losses[-1]

def logging_callback(study, trial):
    elapsed_time = time.time() - start_time
    trial_times.append(elapsed_time)
    
    # Calculate the average time per epoch
    total_epochs = num_epochs
    avg_time_per_epoch = sum(trial_times) / total_epochs
    
    # Estimate the remaining time
    trials_left = n_trials - trial.number - 1
    remaining_epochs = sum(trial.params['num_epochs'] for trial in study.trials[trial.number + 1:])
    estimated_time_left = avg_time_per_epoch * remaining_epochs
    
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Estimated time left: {estimated_time_left:.2f} seconds")

def run_optimization(task='class'):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, task=task), n_trials=n_trials, callbacks=[logging_callback])

    total_time = time.time() - start_time
    print(f"Total optimization time: {total_time:.2f} seconds")

    print('Best trial:')
    trial = study.best_trial
    print(f'  Value: {trial.value}')
    print('  Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')
    print(f'Best trial number: {trial.number}')


# # # # # # # # # # # # #
# Main execution logic  #
# # # # # # # # # # # # #

# Check if CUDA is available
print(f"Is CUDA supported by this system?  {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Set the task
task = 'regression'  # Change this to 'class' for classification, otherwise it will be regression

# Load the dataset
df = pd.read_csv(r'deleteme\predicted_4.9M_dataset.csv')
print(df.head())

# Extract features (columns 1-12) and labels (columns 13-15)
X = df.iloc[:, :12].values
y = df.iloc[:, 12:15].values
print(X[0])
print(y[0])

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

# Run the optimization
num_epochs = 10
n_trials = 100
start_time = time.time()
trial_times = []

run_optimization(task=task)