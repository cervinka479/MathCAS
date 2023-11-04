import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy
#import DataPrep.normalizeDataset as Data
#import DataPrep.inverseNormalizeDataset as Data2
import matplotlib.pyplot as plt

def ANN(hiddenLayers = [12]):
    class NeuralNetwork(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(NeuralNetwork, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    # Define input, hidden, and output sizes
    input_size = 9  # Number of input features (velocity-gradient tensor)
    output_size = 1  # Number of output features (normalized S, Ω, shear tensor)

    # Create the neural network instance
    model = NeuralNetwork(input_size, hiddenLayers[0], output_size)

    # Print the model architecture
    print(model)