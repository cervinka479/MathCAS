import torch.nn as nn

def dynamically_get_module(activation: str) -> nn.Module:
    try:
        activation_class: nn.Module = getattr(nn, activation)
    except AttributeError:
        raise ValueError(f"Activation '{activation}' is not a valid torch.nn module.")
    return activation_class