from torch import nn, optim
from typing import Type

def get_module(module: str) -> Type[nn.Module]:
    try:
        return getattr(nn, module)
    except AttributeError:
        raise ValueError(f"Module '{module}' is not a valid torch.nn module.")
    
def get_loss_function(name: str) -> nn.Module:
    try:
        return getattr(nn, name)()
    except AttributeError:
        raise ValueError(f"Loss function '{name}' is not a valid torch.nn loss.")

def get_optimizer(name: str, params, lr: float) -> optim.Optimizer:
    try:
        return getattr(optim, name)(params, lr=lr)
    except AttributeError:
        raise ValueError(f"Optimizer '{name}' is not a valid torch.optim optimizer.")
    
def get_scheduler(name: str, optimizer, **kwargs):
    if name == "ReduceLROnPlateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    else:
        raise ValueError(f"Scheduler '{name}' is not supported.")