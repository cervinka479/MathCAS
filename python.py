import torch.nn as nn

def ANN(IO=[9,1], hiddenLayers=[12]):
    class NeuralNetwork(nn.Module):
        def __init__(self, hiddenLayers):
            super(NeuralNetwork, self).__init__()
            self.layers = nn.ModuleList()
            input_size = IO[0]
            
            # Create hidden layers
            for hidden_size in hiddenLayers:
                self.layers.append(nn.Linear(input_size, hidden_size))
                self.layers.append(nn.ReLU())
                input_size = hidden_size
            
            # Create output layer
            self.layers.append(nn.Linear(input_size, IO[1]))

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    # Create the neural network instance
    model = NeuralNetwork(hiddenLayers)

    # Print the model architecture
    print(model)

ANN()
