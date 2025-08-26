import os
import sys
from pathlib import Path
from config import load_config
from config.schema import FullConfig
from src.architecture import NeuralNetwork
from torch import nn
from src.trainer import train
from src.evaluation import evaluate
from utils.profiler import enable_profiling, is_profiling_enabled
from utils.experiment import create_experiment_dir, copy_config, log_config_hash
from utils.logger import setup_logger

if __name__ == "__main__":
    enable_profiling(False)
    
    if is_profiling_enabled():
        print("NVTX profiling is enabled")

    # Get template path from command line argument
    if len(sys.argv) < 2:
        print("Usage: python main.py <template_path>")
        sys.exit(1)
    template_path: Path = Path(sys.argv[1])

    config: FullConfig = load_config(template_path)

    # Create experiment directory
    exp_dir: Path = create_experiment_dir(Path(config.output_dir), config.name or "unnamed_experiment")
    copy_config(template_path, exp_dir)

    # Setup logger
    log_file = os.path.join(exp_dir, "nn.log")
    logger = setup_logger(config.verbose, config.save_logs, log_file)

    # Log config hash for reproducibility
    log_config_hash(template_path, logger)

    train(config, exp_dir, logger)

    eval_data_path: Path = Path(r"C:\Users\cervinka\cervinka\GitHub\MathCAS\datasets\dataset_compressible_flow_5M_test.csv")
    evaluate(config, exp_dir, eval_data_path)
