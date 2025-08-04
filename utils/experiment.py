import os
import shutil
import datetime

def create_experiment_dir(output_dir: str, exp_name: str) -> str:
    """
    Create a unique experiment directory with timestamp and experiment name.
    Returns the path to the created directory.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join(output_dir, f"{timestamp}_{exp_name}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

def copy_config(config_path: str, exp_dir: str):
    """
    Copy the config file to the experiment directory for reproducibility.
    """
    try:
        shutil.copy(config_path, os.path.join(exp_dir, "config.yaml"))
    except Exception as e:
        print(f"Warning: Could not copy config file: {e}")
