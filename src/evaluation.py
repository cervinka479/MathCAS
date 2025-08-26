import torch
import pandas as pd
import numpy as np
import time
from pathlib import Path
from config.schema import FullConfig
from src.architecture import NeuralNetwork
from torch.utils.data import DataLoader, TensorDataset
from utils.metrics import compute_regression_metrics
from typing import Any
from types import SimpleNamespace
from torch.utils.data import DataLoader

def load_model(config: Any, exp_dir: Path) -> NeuralNetwork:
    """
    Loads a trained NeuralNetwork model from a specified experiment directory.

    Args:
        config (Any): Configuration object containing model architecture details.
        exp_dir (Path): Path to the experiment directory containing the saved model.

    Returns:
        NeuralNetwork: An instance of NeuralNetwork with loaded weights.

    Raises:
        FileNotFoundError: If the model file does not exist in the specified directory.
        RuntimeError: If there is an error loading the model state dictionary.
    """
    model_path = exp_dir / "best_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = NeuralNetwork(config.architecture)
    try:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
    except Exception as e:
        raise RuntimeError(f"Error loading model state dictionary: {e}")
    return model

def load_data(config: Any, eval_data_path: Path) -> DataLoader:
    """
    Loads evaluation data from a CSV file and returns a DataLoader.

    Args:
        config (Any): Configuration object with data attributes (in_cols, out_cols, batch_size).
        eval_data_path (Path): Path to the evaluation CSV file.

    Returns:
        DataLoader: DataLoader for the evaluation dataset.
    """
    eval_df = pd.read_csv(eval_data_path, usecols=config.data.in_cols + config.data.out_cols)
    X_eval = torch.tensor(eval_df[config.data.in_cols].values, dtype=torch.float32)
    y_eval = torch.tensor(eval_df[config.data.out_cols].values, dtype=torch.float32)

    eval_loader = DataLoader(
        TensorDataset(X_eval, y_eval),
        batch_size=getattr(config.data, "batch_size", 128) or 128,
        shuffle=False
    )
    return eval_loader

def evaluate(config: FullConfig, exp_dir: Path, eval_data_path: Path) -> None:
    model = load_model(config, exp_dir)
    eval_loader = load_data(config, eval_data_path)

    # Set model to eval mode
    model.eval()
    all_preds, all_targets = [], []

    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        for batch_x, batch_y in eval_loader:
            preds = model(batch_x)
            all_preds.append(preds.cpu())
            all_targets.append(batch_y.cpu())
    inference_time = time.time() - start_time

    y_pred = torch.cat(all_preds, dim=0).numpy()
    y_true = torch.cat(all_targets, dim=0).numpy()

    write_results_to_file(config, y_pred, y_true, exp_dir, model, inference_time)

def write_results_to_file(config, y_pred, y_true, exp_dir, model, inference_time):

    output_cols = config.data.out_cols  # list of output column names

    results_lines = []

    # Add experiment info
    experiment_name = exp_dir.name
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    results_lines.append(f"Experiment: {experiment_name}")
    results_lines.append(f"Number of trainable parameters: {num_params}")
    results_lines.append(f"Inference time [s]: {inference_time:.2f}")

    # Save metrics
    output_cols = config.data.out_cols
    for i, col in enumerate(output_cols):
        y_true_col = y_true[:, i]
        y_pred_col = y_pred[:, i]
        metrics = compute_regression_metrics(y_true_col, y_pred_col)
        results_lines.append(f"\nMetrics for output '{col}':")
        for k, v in metrics.items():
            results_lines.append(f"  {k}: {v}")

    # Save the first 10 predictions for each output
    for i, col in enumerate(output_cols):
        results_lines.append(f"\ntrue, prediction for output '{col}':")
        for idx in range(10):
            true_val = np.ravel(y_true[idx])[i]
            pred_val = np.ravel(y_pred[idx])[i]
            results_lines.append(f"{true_val:.5e}, {pred_val:.5e}")

    # Save the 10 worst predictions for each output (by squared error)
    for i, col in enumerate(output_cols):
        y_true_col = y_true[:, i]
        y_pred_col = y_pred[:, i]
        squared_errors = (y_true_col - y_pred_col) ** 2
        worst_indices = np.argsort(squared_errors)[-10:][::-1]  # 10 largest errors, descending
        results_lines.append(f"\ntrue, prediction for output '{col}' (10 worst by squared error):")
        for idx in worst_indices:
            true_val = np.ravel(y_true[idx])[i]
            pred_val = np.ravel(y_pred[idx])[i]
            err = squared_errors[idx]
            results_lines.append(f"{true_val:.5e}, {pred_val:.5e}    # squared error: {err:.5e}")

    # Write to file in experiment folder
    results_path = exp_dir / "evaluation_results.txt"
    with open(results_path, "w") as f:
        for line in results_lines:
            f.write(line + "\n")