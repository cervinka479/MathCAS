import DataPrep

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
        
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
    
    # Create the neural network instance
    model = NeuralNetwork(hl)
    return model

def nnTrain(splitDataset=[["train_input"],["train_output"],["val_input"],["val_output"],["test_input"],["test_output"]], model=nnArch(),optimizer="adam",learningRate=0.01,criterion="mse",batch_size=4, epochs=50, save="", visualize=True):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    # Unpack your dataset
    train_input, train_output, val_input, val_output, test_input, test_output = splitDataset

    # Convert your dataset to PyTorch tensors
    train_data = TensorDataset(torch.tensor(train_input, dtype=torch.float32), torch.tensor(train_output, dtype=torch.float32))
    val_data = TensorDataset(torch.tensor(val_input, dtype=torch.float32), torch.tensor(val_output, dtype=torch.float32))
    test_data = TensorDataset(torch.tensor(test_input, dtype=torch.float32), torch.tensor(test_output, dtype=torch.float32))
    
    # Create a DataLoader for your dataset
    train_loader = DataLoader(train_data, batch_size)
    val_loader = DataLoader(val_data, batch_size)
    test_loader = DataLoader(test_data, batch_size)

    print(model)

    match optimizer:
        case "sgd":
            optimizer = torch.optim.SGD(model.parameters(), learningRate)
        case "adam":
            optimizer = torch.optim.Adam(model.parameters(), learningRate)

    match criterion:
        case "mse":
            criterion=torch.nn.MSELoss()

    train_losses = []
    val_losses = []
    minimal_val_loss = "x"

    # Training loop
    for epoch in range(epochs):
        train_loss = 0
        val_loss = 0
        for inputs, targets in train_loader:
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
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
            val_loss /= len(val_loader)  # Average validation loss

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {loss.item()}, Validation Loss: {val_loss}')

        model.train()  # Set the model back to training mode

        # Save trained model
        if save != "":
            if minimal_val_loss == "x":
                minimal_val_loss = val_loss
            elif val_loss < minimal_val_loss:
                bestModel = model.state_dict()
                minimal_val_loss = val_loss

    print("minimal validation loss: "+str(minimal_val_loss))
    
    if save != "":
        savePath = save+"_VL{"+str("{:.3e}".format(minimal_val_loss))+"}.pth"
        torch.save(bestModel, savePath)
        print("saved model: "+savePath)
    
    # Visualize the training process
    if visualize == True:
        nnVisualize(train_losses, val_losses)

    #return train_losses, val_losses
    return minimal_val_loss

def nnPredict(loadModel, inputDataset, model=nnArch()):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    
    # Load the saved model
    model.load_state_dict(torch.load(loadModel))  # Load the saved parameters
    model.eval()  # Set the model to evaluation mode

    # Convert your input data to a PyTorch tensor
    inputDataset = torch.tensor(inputDataset, dtype=torch.float32)

    # Use the model to make predictions
    with torch.no_grad():
        predictions = model(inputDataset)

    return predictions

def nnVisualize(train_losses,val_losses):
    import matplotlib.pyplot as plt

    # Create a figure
    plt.figure()

    # Plot the training and validation losses for the model
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Show the plot
    plt.show()

# Loss realtion to dataset scale
def valLossComparasion():
    import matplotlib.pyplot as plt
    import numpy as np

    # Create a figure
    plt.figure()

    minimal_val_losses = []
    for i in range(5):
        minimal_val_losses.append(nnTrain(visualize=False, save="XY-5"+str(i+1), splitDataset=DataPrep.split(*DataPrep.extract("dXY-5_1000.csv",limit=200)),model=nnArch(io=[2,1], hl=[16]), epochs=400, learningRate=0.0001))
    for i in range(5):
        minimal_val_losses.append(nnTrain(visualize=False, save="XY-5"+str(i+6), splitDataset=DataPrep.split(*DataPrep.extract("dXY-5_1000.csv",limit=200)),model=nnArch(io=[2,1], hl=[16]), epochs=400, learningRate=0.00001))

    # Plot the validation losses for all models
    plt.plot(minimal_val_losses, label='Minimal Validation Loss')

    plt.xlabel('Iteration index')
    plt.ylabel('Minimal Validation Loss')
    plt.legend()

    # Show the plot
    plt.show()


print(nnPredict(loadModel="lrTest4_VL{1.970e-08}.pth", inputDataset=DataPrep.extract("dPredict.csv",i=[1,2],o=[1,1])[0],model=nnArch(io=[2,1],hl=[16])))

#valLossComparasion()

#nnTrain(save="lrTest4", splitDataset=DataPrep.split(*DataPrep.extract("dXY-5_1000.csv",limit=800)),model=nnArch(io=[2,1], hl=[16]), epochs=1200, learningRate=0.0001)