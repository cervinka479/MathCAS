import torch.nn as nn

def nnArch(io=[9,1], hl=[12]):
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
    print(model)
    return model

# - - - - - - - - - - - - - - - - - - - -

import torch

# Define the size of your dataset
num_samples = 1000

# Create random tensor of size [num_samples x 9]
X = torch.randn(num_samples, 9)

# Create random tensor of size [num_samples x 1] for target
y = torch.randn(num_samples, 1)

# Now you have your data (X) and target labels (y)

# - - - - - - - - - - - - - - - - - - - -

from torch.utils.data import random_split, TensorDataset

# Create a TensorDataset from your data
dataset = TensorDataset(X, y)

# Define the proportions
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

# Split the dataset
train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

# - - - - - - - - - - - - - - - - - - - -

from torch.utils.data import DataLoader

# Define a batch size
batch_size = 32

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

# - - - - - - - - - - - - - - - - - - - -

# Define a loss function and an optimizer
model = nnArch()
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):  # 100 epochs
    for batch in train_loader:
        # Get the data and targets from the batch
        data, targets = batch

        # Forward pass
        output = model(data)
        loss = loss_fn(output, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# - - - - - - - - - - - - - - - - - - - -

# Switch to evaluation mode
model.eval()

# No need to track gradients
with torch.no_grad():
    for batch in test_loader:
        data, targets = batch
        output = model(data)
        loss = loss_fn(output, targets)
        print(f"Test Loss: {loss.item()}")
