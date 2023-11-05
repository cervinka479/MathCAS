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



nnArch(hl=[16,10,8])