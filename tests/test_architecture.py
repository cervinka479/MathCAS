# tests/test_architecture.py
import pytest
from src.architecture import NeuralNetwork
from config.schema import ArchitectureConfig

def test_neural_network_forward():
    config = ArchitectureConfig(
        in_size=9, out_size=3, hidden_layers=[16], activation="ReLU",
        use_dropout=False, dropout=0.0, dropout_inplace=False, final_activation=None
    )
    model = NeuralNetwork(config)
    import torch
    x = torch.randn(2, 9)
    y = model(x)
    assert y.shape == (2, 3)