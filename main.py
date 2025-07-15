from config import load_config
from config.schema import FullConfig
from src.architecture import NeuralNetwork
from torch import nn
from src.trainer import train
from utils.profiler import enable_profiling, is_profiling_enabled

if __name__ == "__main__":
    enable_profiling(True)
    
    if is_profiling_enabled():
        print("NVTX profiling is enabled")

    template_path: str = r'experiments\temporary.yaml'
    train(template_path)