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

import os
import time

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
    
        # Calculate the number of rows for each 1% of the dataset
        one_percent = features_tensor.shape[0] // 100

        start_time = time.time()  # Record the start time

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

            # Print a message every time 1% of the rows have been processed
            if i % one_percent == 0:
                #os.system('cls' if os.name == 'nt' else 'clear')  # Clear the terminal
                elapsed_time = time.time() - start_time  # Calculate the elapsed time
                estimated_time = elapsed_time / (i + 1) * features_tensor.shape[0]  # Estimate the total time
                remaining_time = estimated_time - elapsed_time  # Estimate the remaining time
                print(f'Progress: {i // one_percent}%, Estimated remaining time: {remaining_time} seconds')

    print("Dataset generated: "+new_file_name)

def precise_filter(new_file_name, targetDataset):
    import pandas as pd

    # Load the dataset
    df = pd.read_csv(targetDataset)

    # Filter the DataFrame
    filtered_df = df[df['vorticity'] != 0]

    # Write the filtered DataFrame to a new CSV file
    filtered_df.to_csv(new_file_name, index=False)

# Variables

classModel_Path = "classificator1_VL{2.000e-01}.pth"
classModelArch = nnArchClass(io=[9,1], hl=[32,24])
dataset_Path = "dataset3D10k.csv"
limit = 0

#filter("filtered_datset.csv",classModel_Path,dp.extract(dataset_Path, i=[1,9], o=[10,10], limit=limit),classModelArch)

precise_filter("precise_filtered_datset.csv",dataset_Path)