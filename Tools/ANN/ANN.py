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

def nnArchReg(io=[9,1], hl=[12]):
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


def Predict(loadClassModel, loadRegModel, extractedData, classModel=nnArchClass(), regModel=nnArchReg(), output=False):
    # Load the saved model
    classModel.load_state_dict(torch.load(loadClassModel))  # Load the saved parameters
    regModel.load_state_dict(torch.load(loadRegModel))
    
    classModel.eval()  # Set the model to evaluation mode
    regModel.eval()

    extractedDataCopy = copy.deepcopy(extractedData)
    absmaxScaledData = dp.scale(extractedDataCopy[0],extractedDataCopy[1],method="absmax")

    # Convert your input data to a PyTorch tensor
    class_features_tensor = torch.tensor(extractedData[0], dtype=torch.float32)
    class_labels_tensor = torch.tensor(extractedData[1], dtype=torch.float32)
    reg_features_tensor = torch.tensor(absmaxScaledData[0], dtype=torch.float32)
    reg_labels_tensor = torch.tensor(absmaxScaledData[1], dtype=torch.float32)

    # Initialize a list to store the predictions
    predictions = []

    # Iterate over each row in the dataset
    for i in range(class_features_tensor.shape[0]):
        # Extract the feature for the current row
        class_feature = class_features_tensor[i]
        reg_feature = reg_features_tensor[i]

        # Use the classification model to make a prediction
        with torch.no_grad():
            class_output = classModel(class_feature.unsqueeze(0))
            class_prediction = torch.where(class_output > 0.1, torch.tensor(1.0), torch.tensor(0.0))  # Apply condition to outputs

        # If the class prediction is 1.0, use the regression model to make a prediction
        if class_prediction == 1.0:
            with torch.no_grad():
                reg_output = regModel(reg_feature.unsqueeze(0))
                reg_prediction = reg_output.item()
                inverse_scaler = (max(np.abs(extractedData[0][i,:])))
                reg_prediction = reg_prediction * inverse_scaler
        else:
            reg_prediction = 0.0

        # Append the prediction to the list
        predictions.append(reg_prediction)
        print(reg_prediction)

    return predictions

classModel_Path = "classTest1_VL{1.843e-02}.pth"
classModelArch = nnArchClass(io=[3,1], hl=[32,16])
regModel_Path = "regTest3_VL{3.960e-06}.pth"
regModelArch = nnArchReg(io=[3,1], hl=[32,16])
dataset_Path = "class-test.csv"

print(Predict(loadClassModel=classModel_Path,loadRegModel=regModel_Path,extractedData=dp.extract(path=dataset_Path,i=[1,3],o=[4,4],limit=50),classModel=classModelArch,regModel=regModelArch))