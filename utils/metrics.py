from datetime import timedelta
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def format_time(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))

def compute_regression_metrics(y_true, y_pred):
    """
    Compute regression metrics: MAPE, MAE, MSE, MedAE, MaxAE.
    y_true and y_pred should be numpy arrays of shape (N,) or (N, 1).
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Avoid division by zero in MAPE
    nonzero = y_true != 0
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100) if np.any(nonzero) else float('nan')
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    abs_errors = np.abs(y_true - y_pred)
    medae = float(np.median(abs_errors))
    maxae = float(np.max(abs_errors))

    return {
        "MAPE": f"{mape:.4e}",
        "MAE": f"{mae:.4e}",
        "MSE": f"{mse:.4e}",
        "MedAE": f"{medae:.4e}",
        "MaxAE": f"{maxae:.4e}",
    }