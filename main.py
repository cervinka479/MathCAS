from config import load_config
from config.schema import FullConfig
from src.architecture import NeuralNetwork
from torch import nn
from src.trainer import train
from utils.profiler import enable_profiling, is_profiling_enabled

if __name__ == "__main__":
    enable_profiling(False)
    
    if is_profiling_enabled():
        print("NVTX profiling is enabled")

    template_path: str = r'experiments\eda_shear_50M.yaml'
    train(template_path)