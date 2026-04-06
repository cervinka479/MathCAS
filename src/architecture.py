import torch
import torch.nn as nn
from config.schema import ArchitectureConfig
from utils.modules import get_module



class NeuralNetwork(nn.Module):
    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList()
        
        current_size: int = config.in_size
        for hidden_size in config.hidden_layers:
            self.layers.append(nn.Linear(current_size, hidden_size))
            self.layers.append(get_module(config.activation)())
            if config.use_dropout:
                self.layers.append(nn.Dropout(config.dropout, inplace=config.dropout_inplace))
            current_size = hidden_size
        
        self.layers.append(nn.Linear(current_size, config.out_size))

        if final_activation := config.final_activation:
            self.layers.append(get_module(final_activation)())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
    
    def check_dying_relus(self, data: torch.Tensor, threshold: float = 0.0) -> dict[str, dict]:
        activations: dict[str, torch.Tensor] = {}
        hooks = []

        # Register hooks on ReLU (or LeakyReLU, etc.) layers
        for name, layer in self.layers.named_modules():
            if isinstance(layer, (nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.ELU)):
                def _hook(module, input, output, name=name):
                    activations[name] = output.detach()
                hooks.append(layer.register_forward_hook(_hook))

        # Forward pass
        self.eval()
        with torch.no_grad():
            self.forward(data)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Analyse
        report = {}
        for name, act in activations.items():
            active_frac = (act > 0).float().mean(dim=0)
            dead_mask = active_frac <= threshold
            report[name] = {
                "total": act.shape[1],
                "dead": int(dead_mask.sum().item()),
                "dead_fraction": float(dead_mask.float().mean().item()),
                "dead_indices": dead_mask.nonzero(as_tuple=True)[0].tolist(),
            }

        return report