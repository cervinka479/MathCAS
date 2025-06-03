import torch
import torch.nn as nn
from config.schema import ArchitectureConfig
from utils.modules import dynamically_get_module



class NeuralNetwork(nn.Module):
    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList()
        
        current_size: int = config.in_size
        for hidden_size in config.hidden_layers:
            self.layers.append(nn.Linear(current_size, hidden_size))
            self.layers.append(dynamically_get_module(config.activation)())
            if config.use_dropout:
                self.layers.append(nn.Dropout(config.dropout))
            current_size = hidden_size
        
        self.layers.append(nn.Linear(current_size, config.out_size))

        if final_activation := config.final_activation:
            self.layers.append(dynamically_get_module(final_activation)())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x