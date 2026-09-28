"""Regression evaluation metrics."""
from typing import Dict, Union, Sequence
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(
    y_true: Union[Sequence[float], np.ndarray, pd.Series],
    y_pred: Union[Sequence[float], np.ndarray, pd.Series],
    n_features: int,
) -> Dict[str, float]:
    """Compute MAE, MSE, RMSE, R2, Adjusted R2, and MAPE.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted target values.
        n_features: Number of features used (for adjusted R2 calculation).

    Returns:
        Dict of metric name -> formatted float value.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    n = len(y_true_arr)

    mae = mean_absolute_error(y_true_arr, y_pred_arr)
    mse = mean_squared_error(y_true_arr, y_pred_arr)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_arr, y_pred_arr)
    adj_r2 = 1 - (1 - r2) * (n - 1) / max(n - n_features - 1, 1)
    mape = np.mean(np.abs((y_true_arr - y_pred_arr) / np.clip(np.abs(y_true_arr), 1e-6, None))) * 100

    return {
        "MAE": round(float(mae), 4),
        "MSE": round(float(mse), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4),
        "Adjusted_R2": round(float(adj_r2), 4),
        "MAPE": round(float(mape), 4),
    }
