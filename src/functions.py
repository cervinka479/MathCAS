from config import load_config
from config.schema import ArchitectureConfig
from src.architecture import NeuralNetwork
from typing import Any
from torch import nn

def nn_build(config_path: str) -> nn.Module:
    full_config: dict[str, Any] = load_config(config_path)
    architecture: ArchitectureConfig = full_config['architecture']
    return NeuralNetwork(architecture)

if __name__ == "__main__":
    template_path: str = r'MathCAS_2.0\templates\regression.yaml'
    print(nn_build(template_path))