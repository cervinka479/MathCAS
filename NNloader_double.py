print("A")
import torch
print("A")
import pandas as pd
print("A")
import torch.nn as nn
print("A")
from sklearn.metrics import mean_absolute_error, mean_squared_error
print("A")
import numpy as np
print("A")
import matplotlib.pyplot as plt
print("A")

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
#y_true = df.iloc[:, 12:15].values
y_true = df.iloc[:, 9:11].values
y_shear = df.iloc[:, 11].values

print(X_test[0])
print(y_true[0])
print(y_shear[0])


# Convert to PyTorch tensors
X1_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Load the trained Shear model
shear_model_path = r'best_models\shear_best_model.pth'
dropout_prob = 0.5
task = 'regression'

model = load_model(shear_model_path, dropout_prob, task)

# Make predictions
with torch.no_grad():
    y_shear_pred = model(X1_test_tensor).numpy()

print(X1_test_tensor[0])
print(y_shear_pred[0])

# # # # #

print(y_shear_pred.shape)
print(y_shear.shape)

# Combine X_test with y_shear_pred to create the input for the residual model
X2_test = np.hstack((X_test, y_shear_pred))
#X2_test = np.hstack((X_test, y_shear))
X2_test_tensor = torch.tensor(X2_test, dtype=torch.float32)

# Load the trained Residual model
residual_model_path = r'best_models\residual_true_shear.pth'
dropout_prob = 0.5
task = 'regression'

model = load_model(residual_model_path, dropout_prob, task)

# Make predictions
with torch.no_grad():
    y_residual_pred = model(X2_test_tensor).numpy()

print(X2_test_tensor[0])
print(y_residual_pred[0])

y_pred = np.hstack((y_residual_pred, y_shear_pred))
print(y_pred[0])

# # # # #

correction = False

if correction:

    # Combine X_test with y_pred to create the input for the correction model
    X3_test = np.hstack((X_test, y_pred))
    X3_test_tensor = torch.tensor(X3_test, dtype=torch.float32)

    # Load the trained Correction model
    correction_model_path = r'correction_best_model.pth'
    dropout_prob = 0.5
    task = 'regression'

    model = load_model(correction_model_path, dropout_prob, task)

    # Make predictions
    with torch.no_grad():
        y_pred = model(X3_test_tensor).numpy()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

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
chunk_size = 1000  # Adjust chunk size as needed
abs_errors = []
for i in range(0, len(y_true[:,0]), chunk_size):
    chunk_abs_errors = np.abs(y_true[i:i + chunk_size, 0] - y_pred[i:i + chunk_size, 0])
    abs_errors.extend(chunk_abs_errors)

v_medae = np.median(abs_errors)  # Median Absolute Error
v_maxae = np.max(abs_errors)  # Max Absolute Error

print(f"Mean Absolute Percentage Error (MAPE): {v_mape}")
print(f"Mean Squared Error (MSE): {v_mse}")
print(f"Mean Absolute Error (MAE): {v_mae}")
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
print(f"Mean Squared Error (MSE): {sr_mse}")
print(f"Mean Absolute Error (MAE): {sr_mae}")
print(f"Median Absolute Error (MedAE): {sr_medae}")
print(f"Max Absolute Error (MaxAE): {sr_maxae}")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# Constants
v_tmax = np.sqrt(6)
sr_tmax = np.sqrt(8)

# Create a figure with two subplots (1 row, 2 columns)
subset_size = 50000  # Adjust the number of points to plot
indices = np.random.choice(len(y_true), subset_size, replace=False)  # Randomly sample indices
fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Adjust figure size as needed

# First subplot: y_true[:, 0] vs. y_pred[:, 0] (using v_tmax)
axes[0].scatter(y_true[indices, 0], y_pred[indices, 0], color='blue', s=1, label='Predicted vs True')
axes[0].plot([0, v_tmax], [0, v_tmax], color='red', linestyle='-')  # Ideal line
axes[0].plot([0+v_tmax/100, v_tmax+v_tmax/100], [0, v_tmax], color='red', alpha=0.75, linestyle='--')
axes[0].plot([0, v_tmax], [0+v_tmax/100, v_tmax+v_tmax/100], color='red', alpha=0.75, linestyle='--')
axes[0].set_xlabel('True Values')
axes[0].set_ylabel('Predicted Values')
axes[0].set_title(residual_model_path+" (Residual Vorticity)")
axes[0].legend()
axes[0].grid(True)

# Second subplot: y_true[:, 1] vs. y_pred[:, 1] (using sr_tmax)
axes[1].scatter(y_true[indices, 1], y_pred[indices, 1], color='green', s=1, label='Predicted vs True')
axes[1].plot([0, sr_tmax], [0, sr_tmax], color='red', linestyle='-')  # Ideal line
axes[1].plot([0+sr_tmax/100, sr_tmax+sr_tmax/100], [0, sr_tmax], color='red', alpha=0.75, linestyle='--')
axes[1].plot([0, sr_tmax], [0+sr_tmax/100, sr_tmax+sr_tmax/100], color='red', alpha=0.75, linestyle='--')
axes[1].set_xlabel('True Values')
axes[1].set_ylabel('Predicted Values')
axes[1].set_title(residual_model_path+" (Residual Strain-Rate)")
axes[1].legend()
axes[1].grid(True)

# Adjust layout and show the plot
plt.tight_layout()
plt.savefig(residual_model_path+" (Residual Predicted vs True).pdf")
plt.show()

# # # # # # # # # #

fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Adjust figure size as needed

# Plot true values vs. absolute errors
absolute_errors = [np.abs(y_true[indices, 0] - y_pred[indices, 0].flatten()), np.abs(y_true[indices, 1] - y_pred[indices, 1].flatten())]

axes[0].scatter(y_true[indices, 0], absolute_errors[0], color='purple', s=1, alpha=0.75, label='Absolute Error vs True Value')  # Make dots smaller with s=1

axes[0].set_xlabel('True Values')
axes[0].set_ylabel('Absolute Error')
axes[0].set_title(residual_model_path+" (Residual Vorticity)")
axes[0].legend()
axes[0].grid(True)

axes[1].scatter(y_true[indices, 1], absolute_errors[1], color='olive', s=1, alpha=0.75, label='Absolute Error vs True Value')  # Make dots smaller with s=1

axes[1].set_xlabel('True Values')
axes[1].set_ylabel('Absolute Error')
axes[1].set_title(residual_model_path+" (Residual Strain-Rate)")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(residual_model_path+" (Residual Absolute Error).pdf")
plt.show()

# # Combine X_test, y_pred, and y_true into a single DataFrame
# results_df = pd.DataFrame(np.hstack((X_test, y_pred, y_true)),
#                           columns=[f'X{i+1}' for i in range(X_test.shape[1])] +
#                                   [f'y_pred{i+1}' for i in range(y_pred.shape[1])] +
#                                   [f'y_true{i+1}' for i in range(y_true.shape[1])])

# # Save the DataFrame to a CSV file
# results_df.to_csv(r'results.csv', index=False)
