import DataPrep as dp
import torch
import pandas as pd
import copy
import numpy as np

def nnArchClass(io=[9,1], hl=[12]):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
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

    import torch.nn as nn
    
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
        
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
    
    # Create the neural network instance
    model = NeuralNetwork(hl)
    return model

def filter(new_file_name, loadClassModel, extractedData, classModel=nnArchClass()):
    # Load the saved model
    if torch.cuda.is_available():
        classModel.load_state_dict(torch.load(loadClassModel))  # Load the saved parameters
    else:
        classModel.load_state_dict(torch.load(loadClassModel,map_location=torch.device('cpu')))
    
    classModel.eval()  # Set the model to evaluation mode

    extractedDataCopy = copy.deepcopy(extractedData)
    absmaxScaledData = dp.scale(extractedDataCopy[0],extractedDataCopy[1],method="absmax")

    # Convert your input data to a PyTorch tensor
    features_tensor = torch.tensor(absmaxScaledData[0], dtype=torch.float32)

    import csv

    with open(new_file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["A11","A12","A13","A21","A22","A23","A31","A32","A33","vorticity"])
    
        # Iterate over each row in the dataset
        for i in range(features_tensor.shape[0]):
            # Extract the feature for the current row
            class_feature = features_tensor[i]

            # Use the classification model to make a prediction
            with torch.no_grad():
                class_output = classModel(class_feature.unsqueeze(0))
                class_prediction = torch.where(class_output > 0.5, torch.tensor(1.0), torch.tensor(0.0))  # Apply condition to outputs

            # If the class prediction is 1.0, use the regression model to make a prediction
            if class_prediction == 1.0:
                writer.writerow([*extractedData[0][i],*extractedData[1][i]])

    print("Dataset generated: "+new_file_name)

# Variables

classModel_Path = "classificator1_VL{2.000e-01}.pth"
classModelArch = nnArchClass(io=[9,1], hl=[32,24])
dataset_Path = "dataset3D10k.csv"
limit = 0

filter("filtered_datset.csv",classModel_Path,dp.extract(dataset_Path, i=[1,9], o=[10,10], limit=limit),classModelArch)