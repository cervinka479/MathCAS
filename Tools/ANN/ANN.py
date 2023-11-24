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

def nnTrainOld(splitDataset=[["train_input"],["train_output"],["val_input"],["val_output"],["test_input"],["test_output"]], model=nnArch(),optimizer="adam",learningRate=0.01,criterion="mse", epochs=50, save=""):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    # Unpack your dataset
    train_input, train_output, val_input, val_output, test_input, test_output = splitDataset

    # Convert your dataset to PyTorch tensors
    train_data = TensorDataset(torch.tensor(train_input, dtype=torch.float32), torch.tensor(train_output, dtype=torch.float32))
    val_data = TensorDataset(torch.tensor(val_input, dtype=torch.float32), torch.tensor(val_output, dtype=torch.float32))
    test_data = TensorDataset(torch.tensor(test_input, dtype=torch.float32), torch.tensor(test_output, dtype=torch.float32))
    
    # Create a DataLoader for your dataset
    train_loader = DataLoader(train_data, batch_size=5)
    val_loader = DataLoader(val_data, batch_size=5)
    test_loader = DataLoader(test_data, batch_size=5)

    print(model)

    match optimizer:
        case "sgd":
            optimizer = torch.optim.SGD(model.parameters(), learningRate)
        case "adam":
            optimizer = torch.optim.Adam(model.parameters(), learningRate)

    match criterion:
        case "mse":
            criterion=torch.nn.MSELoss()

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

        # Print loss for every epoch
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

    # Evaluation
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            print(f'Final Loss: {loss.item()}')

    # Save trained model
    if save != "":
        torch.save(model.state_dict(), save+".pth")
        print("saved model: "+save+".pth")

def nnTrainVal(splitDataset=[["train_input"],["train_output"],["val_input"],["val_output"],["test_input"],["test_output"]], model=nnArch(),optimizer="adam",learningRate=0.01,criterion="mse", epochs=50, save=""):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    # Unpack your dataset
    train_input, train_output, val_input, val_output, test_input, test_output = splitDataset

    # Convert your dataset to PyTorch tensors
    train_data = TensorDataset(torch.tensor(train_input, dtype=torch.float32), torch.tensor(train_output, dtype=torch.float32))
    val_data = TensorDataset(torch.tensor(val_input, dtype=torch.float32), torch.tensor(val_output, dtype=torch.float32))
    test_data = TensorDataset(torch.tensor(test_input, dtype=torch.float32), torch.tensor(test_output, dtype=torch.float32))
    
    # Create a DataLoader for your dataset
    train_loader = DataLoader(train_data, batch_size=5)
    val_loader = DataLoader(val_data, batch_size=5)
    test_loader = DataLoader(test_data, batch_size=5)

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

    # Evaluation
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            print(f'Final Loss: {loss.item()}')

    # Save trained model
    if save != "":
        torch.save(model.state_dict(), save+".pth")
        print("saved model: "+save+".pth")

    return train_losses, val_losses

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

def nnTrainVisual():
    import matplotlib.pyplot as plt

    # Train your model and get the losses
    train_losses1, val_losses1 = nnTrainVal(splitDataset=DataPrep.split(*DataPrep.extract("datasetSum.csv")),model=nnArch(io=[2,1]), epochs=200, learningRate=0.01)
    train_losses2, val_losses2 = nnTrainVal(splitDataset=DataPrep.split(*DataPrep.extract("datasetSum.csv")),model=nnArch(io=[2,1]), epochs=200, learningRate=0.001)

    # Create a figure
    plt.figure()

    # Plot the training and validation losses for model 1
    plt.plot(train_losses1, label='Training Loss - Model 1')
    plt.plot(val_losses1, label='Validation Loss - Model 1')

    # Plot the training and validation losses for model 2
    plt.plot(train_losses2, label='Training Loss - Model 2')
    plt.plot(val_losses2, label='Validation Loss - Model 2')

    plt.title('Model Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Show the plot
    plt.show()


#nnTrainVal(save="test1", splitDataset=DataPrep.split(*DataPrep.extract("datasetSum.csv")),model=nnArch(io=[2,1]), epochs=200, learningRate=0.001)

#print(nnPredict(loadModel="testModelSum.pth", inputDataset=DataPrep.extract("datasetPredict.csv",i=[1,2],o=[1,1])[0],model=nnArch(io=[2,1])))

#nnTrainVal(splitDataset=DataPrep.split(*DataPrep.extract("datasetSum.csv")),model=nnArch(io=[2,1]), epochs=200, learningRate=0.0001)

nnTrainVisual()