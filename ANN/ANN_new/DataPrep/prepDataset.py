import numpy as np
import torch


# Load the dataset from .dat file
dataset = np.loadtxt("ANN_DataSet\sphere_Re300_training_nstep180.dat")

# Separate input features (velocity-gradient tensors) and output values (normalized tensors)
input_features = dataset[3, :12]
output_values = dataset[:, 12:]

# Define a function for min-max normalization
def min_max_normalize(data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data

# Normalize input features
input_features_normalized = min_max_normalize(input_features)

# Normalize output values to the same range as the network's output layer activations
output_values_normalized = min_max_normalize(output_values)

# Now input_features_normalized and output_values_normalized can be used for training the neural network
