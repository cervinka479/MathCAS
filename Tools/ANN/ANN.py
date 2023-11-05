import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy
#import DataPrep.normalizeDataset as Data
#import DataPrep.inverseNormalizeDataset as Data2
import matplotlib.pyplot as plt

def ANN(IO=[9,2],hiddenLayers=[12]):
    class NeuralNetwork(nn.Module):
        def __init__(self, hidden_size):
            super(NeuralNetwork, self).__init__()
            self.fc1 = nn.Linear(IO[0], hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, IO[1])

    # Create the neural network instance
    model = NeuralNetwork(hiddenLayers[0])

    # Print the model architecture
    print(model)

ANN(hiddenLayers=[16])