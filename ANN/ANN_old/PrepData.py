import numpy as np
import torch



# Read the .dat file
file_path1 = 'training.dat'  # Replace with the actual file path
data1 = np.genfromtxt(file_path1, delimiter=',', skip_header=1)

# Extract input data1 (3x3 matrices) and labels (coordinates and the value F)
features = data1[:, 3:12]  # Extract the 9 input values (A11 to A33)
labels = data1[:, 12:16]  # Extract the 4 output values (Alpha, Beta, Gamma, Goal)

features = torch.tensor(features, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.float32)



# Read the .dat file
file_path2 = 'evaluation.dat'  # Replace with the actual file path
data2 = np.genfromtxt(file_path2, delimiter=',', skip_header=1)

# Extract input data2 (3x3 matrices) and labels (coordinates and the value F)
test_features = data2[:, 3:12]  # Extract the 9 input values (A11 to A33)

test_features = torch.tensor(test_features, dtype=torch.float32)