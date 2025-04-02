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
dataset_path = r'deleteme\dataset_compressible_flow_500K_test_nstep180.csv'
df = pd.read_csv(dataset_path)

# Extract the first 10 data points
X_test = df.iloc[:, :9].values
y_true = df.iloc[:, 11].values
y_all = df.iloc[:, 9:11].values

# Convert to PyTorch tensors
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Load the trained model
model_path = r'best_models\.shear_best_model.pth'  # Use raw string to handle backslashes
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
chunk_size = 1000  # Adjust chunk size as needed
abs_errors = []
for i in range(0, len(y_true), chunk_size):
    chunk_abs_errors = np.abs(y_true[i:i + chunk_size] - y_pred[i:i + chunk_size])
    abs_errors.extend(chunk_abs_errors)

medae = np.median(abs_errors)  # Median Absolute Error
maxae = np.max(abs_errors)  # Max Absolute Error

print(f"Mean Absolute Percentage Error (MAPE): {mape}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Median Absolute Error (MedAE): {medae}")
print(f"Max Absolute Error (MaxAE): {maxae}")

# Plot true values vs. predicted values
subset_size = 50000  # Adjust the number of points to plot
indices = np.random.choice(len(y_true), subset_size, replace=False)  # Randomly sample indices
plt.figure(figsize=(6, 6))
plt.scatter(y_true[indices], y_pred[indices], color='blue', s=1, alpha=0.75, label='Predicted vs True')  # Make dots smaller with s=1
tmax = np.sqrt(8)

plt.plot([0, tmax], [0, tmax], color='red', linestyle='-') # Ideal line
plt.plot([0+tmax/100, tmax+tmax/100], [0, tmax], color='red', alpha=0.75, linestyle='--')
plt.plot([0, tmax], [0+tmax/100, tmax+tmax/100], color='red', alpha=0.75, linestyle='--')

plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title(model_path)
plt.legend()
plt.grid(True)
plt.savefig(model_path+" (Shear Predicted vs True).pdf")
plt.show()

# Plot true values vs. absolute errors
absolute_errors = np.abs(y_true[indices] - y_pred[indices].flatten())
plt.figure(figsize=(6, 6))
plt.scatter(y_true[indices], absolute_errors, color='purple', s=1, alpha=0.75, label='Absolute Error vs True Value')  # Make dots smaller with s=1

plt.xlabel('True Values')
plt.ylabel('Absolute Error')
plt.title(model_path)
plt.legend()
plt.grid(True)
plt.savefig(model_path+" (Shear Absolute Error).pdf")
plt.show()

# # Combine X_test, y_pred, and y_true into a single DataFrame
# y_pred_reshaped = y_pred.reshape(-1, 1)  # Reshape y_pred to 2D

# shear_results_df = pd.DataFrame(np.hstack((X_test, y_all, y_pred_reshaped)),
#                           columns=[f'X{i+1}' for i in range(X_test.shape[1])] +
#                                   [f'y_true{i+1}' for i in range(y_all.shape[1])] +
#                                   [f'shear_pred' for i in range(y_pred_reshaped.shape[1])])

# # Save the DataFrame to a CSV file
# shear_results_df.to_csv(r'deleteme\dataset_compressible_flow_5M_B_shear_predicted.csv', index=False)