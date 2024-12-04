############################################################################

import numpy as np
import random

def generate():
    Ux = random.random()*2-1
    Uy = random.random()*2-1
    Vx = random.random()*2-1

    s = np.sqrt(4*(Ux**2)+(Uy+Vx)**2)/2
    ω = (Vx-Uy)/2

    if np.abs(s) <= np.abs(ω):
        ωRES = np.sign(ω)*(np.abs(ω)-np.abs(s))
    else:
        ωRES = 0

    return Ux,Uy,Vx,ωRES

############################################################################

def nnArch(io=[9,1], hl=[12]):
    import torch.nn as nn
    
    class NeuralNetwork(nn.Module):
        def __init__(self, hl):
            super(NeuralNetwork, self).__init__()
            self.layers = nn.ModuleList()
            input_size = io[0]
            
            # Create input layer and hidden layers
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

############################################################################

# Create a DataLoader for training and validation subsets
train_loader = DataLoader(train_data, batch_size)
val_loader = DataLoader(val_data, batch_size)

############################################################################

model.train()  # Set the model to training mode
for inputs, targets in train_loader:

    # Forward pass
    outputs = model(inputs)
    
    # Calculate the loss
    loss = criterion(outputs, targets)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    
    # Update the weights and bias
    optimizer.step()
    
    train_loss += loss.item()

train_loss /= len(train_loader) # Average training loss

############################################################################

model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    for inputs, targets in val_loader:

        # Forward pass
        outputs = model(inputs)
        
        # Calculate the loss
        loss = criterion(outputs, targets)
        
        val_loss += loss.item()
    
    val_loss /= len(val_loader)  # Average validation loss

############################################################################