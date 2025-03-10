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

# Extract the first 10 data points
X_test = df.iloc[:, :9].values
y_true = df.iloc[:, 14].values

# Convert to PyTorch tensors
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Load the trained model
model_path = r'best_models\shear_best_model.pth'  # Use raw string to handle backslashes
dropout_prob = 0.5
task = 'regression'

model = load_model(model_path, dropout_prob, task)

# Make predictions
with torch.no_grad():
    y_pred = model(X_test_tensor).numpy()

# Print true results and predictions
print("True results:", y_true[:10])
print("Predictions:", y_pred[:10].flatten())

# Calculate accuracy metrics
list_of_mape_errors = []
for i in range(len(y_true)):
    list_of_mape_errors.append(np.abs((y_true[i] - y_pred[i]) / y_true[i]) * 100)

mape = np.mean(list_of_mape_errors)
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)

# Compute Median Absolute Error in a memory-efficient way
chunk_size = 10000  # Adjust chunk size as needed
abs_errors = []
for i in range(0, len(y_true), chunk_size):
    chunk_abs_errors = np.abs(y_true[i:i + chunk_size] - y_pred[i:i + chunk_size])
    abs_errors.extend(chunk_abs_errors)

medae = np.median(abs_errors)  # Median Absolute Error
maxae = np.max(abs_errors)  # Max Absolute Error

print(f"Mean Absolute Percentage Error (MAPE): {mape}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Median Absolute Error (MedAE): {medae}")
print(f"Max Absolute Error (MaxAE): {maxae}")

# Plot true values vs. predicted values
plt.figure(figsize=(6, 6))
plt.scatter(y_true, y_pred, color='blue', s=1, label='Predicted vs True')  # Make dots smaller with s=1
tmax = np.sqrt(8)

plt.plot([0, tmax], [0, tmax], color='red', linestyle='-') # Ideal line
plt.plot([0+tmax/100, tmax+tmax/100], [0, tmax], color='red', alpha=0.75, linestyle='--')
plt.plot([0, tmax], [0+tmax/100, tmax+tmax/100], color='red', alpha=0.75, linestyle='--')

plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title(model_path)
plt.legend()
plt.grid(True)
plt.show()