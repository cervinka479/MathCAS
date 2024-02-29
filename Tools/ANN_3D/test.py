import torch

# Check if CUDA is available
print(f"Is CUDA supported by this system?  {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def nnArch(io=[9,1], hl=[12]):
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
            self.layers.append(nn.Sigmoid())
        
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
    
    # Create the neural network instance
    model = NeuralNetwork(hl)
    return model

def nnTrain(splitDataset=[["train_input"],["train_output"],["val_input"],["val_output"]], model=nnArch(),optimizer="adam",learningRate=0.01,criterion="bce",batch_size=32, epochs=50, save="test", visualize=True, cli=True):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    import copy

    # Unpack your dataset
    train_input, train_output, val_input, val_output = splitDataset

    # Convert your dataset to PyTorch tensors
    train_data = TensorDataset(torch.tensor(train_input, dtype=torch.float32), torch.tensor(train_output, dtype=torch.float32))
    val_data = TensorDataset(torch.tensor(val_input, dtype=torch.float32), torch.tensor(val_output, dtype=torch.float32))
    
    # Create a DataLoader for your dataset
    train_loader = DataLoader(train_data, batch_size, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size, pin_memory=True)

    print(model)
    model.to(device)

    match optimizer:
        case "sgd":
            optimizer = torch.optim.SGD(model.parameters(), learningRate)
        case "adam":
            optimizer = torch.optim.Adam(model.parameters(), learningRate)

    match criterion:
        case "mse":
            criterion=torch.nn.MSELoss()
        case "bce":
            criterion=torch.nn.BCELoss()

    train_losses = []
    val_losses = []
    val_accuracy = []
    models = []
    minimal_val_loss = "x"

    for epoch in range(epochs):
            train_loss = 0
            val_loss = 0
            correct = 0
            total = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation at the end of the epoch
            model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    # Calculate accuracy
                    predicted = torch.round(outputs.data)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                val_loss /= len(val_loader)  # Average validation loss
                val_acc = correct / total
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracy.append(val_acc)         
            
            print(f'Epoch {epoch+1}/{epochs}, Training Loss: {train_loss}, Validation Loss: {val_loss}, Validation Accuracy: {val_acc}')
            
            model.train()  # Set the model back to training mode
            
            models.append(copy.deepcopy(model.state_dict()))
            
            # Save trained model
            if minimal_val_loss == "x":
                minimal_val_loss = val_loss
            elif val_loss < minimal_val_loss:
                bestModel = model.state_dict()
                minimal_val_loss = val_loss
        
    print("minimal validation loss: "+str(minimal_val_loss))

import DataPrep

modelArchitecture = nnArch(io=[9,1], hl=[48,32])


# Trainig section
import copy
extractedData = DataPrep.extract(path="bin-dataset3D10k.csv",i=[1,9],o=[10,10],limit=0)
extractedDataCopy = copy.deepcopy(extractedData)
absmaxScaledData = DataPrep.scale(extractedDataCopy[0],extractedDataCopy[1],method="absmax",class_labels=True)

#nnTrain(save="classTest",splitDataset=DataPrep.split(*extractedData),model=modelArchitecture, epochs=50, learningRate=0.001, batch_size=32)
nnTrain(save="classTestNorm",splitDataset=DataPrep.split(*absmaxScaledData),model=modelArchitecture, epochs=50, learningRate=0.001, batch_size=32)