from config import load_config
from config.schema import FullConfig
from src.architecture import NeuralNetwork
from torch import nn
from src.trainer import train

if __name__ == "__main__":
    template_path: str = r'templates\regression.yaml'
    train(template_path)