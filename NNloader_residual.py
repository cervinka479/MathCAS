import torch
import pandas as pd
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Define the model architecture
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, dropout_prob=0.5, task='regression'):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        
        # Create hidden layers with dropout
        for hidden_size in hidden_layers:
            self.layers.append(nn.Linear(input_size, hidden_size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_prob))
            input_size = hidden_size
        
        # Create output layer
        self.layers.append(nn.Linear(input_size, output_size))
        if task == 'class':
            self.layers.append(nn.Sigmoid())
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def infer_architecture(state_dict):
    input_size = None
    hidden_layers = []
    output_size = None

    for key in state_dict.keys():
        if 'weight' in key:
            layer_shape = state_dict[key].shape
            if input_size is None:
                input_size = layer_shape[1]
            hidden_layers.append(layer_shape[0])
            output_size = layer_shape[0]

    # Remove the last hidden layer size as it is the output layer size
    hidden_layers = hidden_layers[:-1]

    return input_size, hidden_layers, output_size

def load_model(model_path, dropout_prob=0.5, task='regression'):
    # Load the state dictionary
    state_dict = torch.load(model_path)
    
    # Infer the architecture
    input_size, hidden_layers, output_size = infer_architecture(state_dict)
    print(f"Inferred architecture: Input size = {input_size}, Hidden layers = {hidden_layers}, Output size = {output_size}")

    # Create the model
    model = NeuralNetwork(input_size, hidden_layers, output_size, dropout_prob, task)
    model.load_state_dict(state_dict)
    model.eval()  # Set the model to evaluation mode
    return model

# Load the dataset
dataset_path = r'deleteme\test_subset_100K.csv'
df = pd.read_csv(dataset_path)

# Extract the first 10 data points with features (columns 1-9 and 15) and labels (columns 13 and 14)
X_test = df.iloc[:, list(range(9)) + [14]].values
#X_test = df.iloc[:, :9].values
y_true = df.iloc[:, 12:14].values

# Convert to PyTorch tensors
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Load the trained model
model_path = r'residual_best_model_4x256.pth'  # Use raw string to handle backslashes
dropout_prob = 0.5
task = 'regression'

model = load_model(model_path, dropout_prob, task)

# Make predictions
with torch.no_grad():
    y_pred = model(X_test_tensor).numpy()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# Print true results and predictions
print("True vorticity:", y_true[:10,0])
print("Predicted vorticity:", y_pred[:10,0].flatten())

# Calculate accuracy metrics
list_of_v_mape_errors = []
for i in range(len(y_true[:,0])):
    if y_true[i, 0] != 0:
        list_of_v_mape_errors.append(np.abs((y_true[i,0] - y_pred[i,0]) / y_true[i,0]) * 100)

v_mape = np.mean(list_of_v_mape_errors)
v_mae = mean_absolute_error(y_true[:,0], y_pred[:,0])
v_mse = mean_squared_error(y_true[:,0], y_pred[:,0])

# Compute Median Absolute Error in a memory-efficient way
chunk_size = 10000  # Adjust chunk size as needed
abs_errors = []
for i in range(0, len(y_true[:,0]), chunk_size):
    chunk_abs_errors = np.abs(y_true[i:i + chunk_size, 0] - y_pred[i:i + chunk_size, 0])
    abs_errors.extend(chunk_abs_errors)

v_medae = np.median(abs_errors)  # Median Absolute Error
v_maxae = np.max(abs_errors)  # Max Absolute Error

print(f"Mean Absolute Percentage Error (MAPE): {v_mape}")
print(f"Mean Absolute Error (MAE): {v_mae}")
print(f"Mean Squared Error (MSE): {v_mse}")
print(f"Median Absolute Error (MedAE): {v_medae}")
print(f"Max Absolute Error (MaxAE): {v_maxae}")

print()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

print("True strain-rate:", y_true[:10,1])
print("Predicted strain-rate:", y_pred[:10,1].flatten())

# Calculate accuracy metrics
list_of_sr_mape_errors = []
for i in range(len(y_true[:,1])):
    list_of_sr_mape_errors.append(np.abs((y_true[i,1] - y_pred[i,1]) / y_true[i,1]) * 100)

sr_mape = np.mean(list_of_sr_mape_errors)
sr_mae = mean_absolute_error(y_true[:,1], y_pred[:,1])
sr_mse = mean_squared_error(y_true[:,1], y_pred[:,1])

# Compute Median Absolute Error in a memory-efficient way
chunk_size = 10000  # Adjust chunk size as needed
abs_errors = []
for i in range(0, len(y_true[:,1]), chunk_size):
    chunk_abs_errors = np.abs(y_true[i:i + chunk_size, 1] - y_pred[i:i + chunk_size, 1])
    abs_errors.extend(chunk_abs_errors)

sr_medae = np.median(abs_errors)  # Median Absolute Error
sr_maxae = np.max(abs_errors)  # Max Absolute Error

print(f"Mean Absolute Percentage Error (MAPE): {sr_mape}")
print(f"Mean Absolute Error (MAE): {sr_mae}")
print(f"Mean Squared Error (MSE): {sr_mse}")
print(f"Median Absolute Error (MedAE): {sr_medae}")
print(f"Max Absolute Error (MaxAE): {sr_maxae}")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# Constants
v_tmax = np.sqrt(6)
sr_tmax = np.sqrt(8)

# Create a figure with two subplots (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Adjust figure size as needed

# First subplot: y_true[:, 0] vs. y_pred[:, 0] (using v_tmax)
axes[0].scatter(y_true[:, 0], y_pred[:, 0], color='blue', s=1, label='Predicted vs True')
axes[0].plot([0, v_tmax], [0, v_tmax], color='red', linestyle='-')  # Ideal line
axes[0].plot([0+v_tmax/100, v_tmax+v_tmax/100], [0, v_tmax], color='red', alpha=0.75, linestyle='--')
axes[0].plot([0, v_tmax], [0+v_tmax/100, v_tmax+v_tmax/100], color='red', alpha=0.75, linestyle='--')
axes[0].set_xlabel('True Values')
axes[0].set_ylabel('Predicted Values')
axes[0].set_title(model_path+" (Residual Vorticity)")
axes[0].legend()
axes[0].grid(True)

# Second subplot: y_true[:, 1] vs. y_pred[:, 1] (using sr_tmax)
axes[1].scatter(y_true[:, 1], y_pred[:, 1], color='green', s=1, label='Predicted vs True')
axes[1].plot([0, sr_tmax], [0, sr_tmax], color='red', linestyle='-')  # Ideal line
axes[1].plot([0+sr_tmax/100, sr_tmax+sr_tmax/100], [0, sr_tmax], color='red', alpha=0.75, linestyle='--')
axes[1].plot([0, sr_tmax], [0+sr_tmax/100, sr_tmax+sr_tmax/100], color='red', alpha=0.75, linestyle='--')
axes[1].set_xlabel('True Values')
axes[1].set_ylabel('Predicted Values')
axes[1].set_title(model_path+" (Residual Strain-Rate)")
axes[1].legend()
axes[1].grid(True)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()