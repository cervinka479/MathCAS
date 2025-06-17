from config import load_config
from config.schema import FullConfig
from src.architecture import NeuralNetwork
from torch import nn
from src.dataset import prepare_dataloaders



def nn_build(config_path: str) -> nn.Module:
    config: FullConfig = load_config(config_path)
    return NeuralNetwork(config.architecture)

if __name__ == "__main__":
    template_path: str = r'templates\experiment1.yaml'
    
    print(nn_build(template_path))

    train_loader, val_loader = prepare_dataloaders(template_path)

    # Preview the first batch to confirm
    for batch_x, batch_y in train_loader:
        print("Sample batch X:", batch_x)
        print("Sample batch Y:", batch_y)
        break