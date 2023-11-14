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

def nnTrain(splitDataset=[["train_input"],["train_output"],["val_input"],["val_output"],["test_input"],["test_output"]], model=nnArch(),optimizer="adam",learningRate=0.01,criterion="mse", epochs=50):
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

nnTrain(splitDataset=DataPrep.split(*DataPrep.extract("test.csv")),model=nnArch(io=[2,1]), epochs=200)