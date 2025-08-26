import os
import shutil
import datetime
import hashlib
from pathlib import Path

def create_experiment_dir(output_dir: Path, exp_name: str) -> Path:
    """
    Create a unique experiment directory with timestamp and experiment name.
    Returns the path to the created directory.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = output_dir / f"{timestamp}_{exp_name}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir

def copy_config(config_path: Path, exp_dir: Path):
    """
    Copy the config file to the experiment directory for reproducibility.
    """
    try:
        shutil.copy(config_path, exp_dir / "config.yaml")
    except Exception as e:
        print(f"Warning: Could not copy config file: {e}")

def log_config_hash(config_path: Path, logger):
    """
    Log config hash for reproducibility.
    """
    try:
        with open(config_path, 'rb') as f:
            config_hash = hashlib.md5(f.read()).hexdigest()
        logger.info(f"Config hash: {config_hash}")
    except Exception as e:
        logger.warning(f"Could not compute config hash: {e}")