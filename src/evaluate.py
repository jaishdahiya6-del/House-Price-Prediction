"""Regression evaluation metrics."""
from typing import Dict
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true, y_pred, n_features: int) -> Dict[str, float]:
    """Compute MAE, MSE, RMSE, R2, Adjusted R2, and MAPE.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted target values.
        n_features: Number of features used (for adjusted R2).

    Returns:
        Dict of metric name -> value.
    """
    n = len(y_true)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / max(n - n_features - 1, 1)
    mape = np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / np.clip(np.abs(y_true), 1e-6, None))) * 100

    return {
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "Adjusted_R2": round(adj_r2, 4),
        "MAPE": round(mape, 4),
    }
