"""Unit tests for evaluation metrics module."""
import numpy as np
from src.evaluate import compute_metrics


def test_compute_metrics():
    y_true = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    y_pred = np.array([110.0, 190.0, 310.0, 390.0, 510.0])

    metrics = compute_metrics(y_true, y_pred, n_features=2)
    assert "MAE" in metrics
    assert "MSE" in metrics
    assert "RMSE" in metrics
    assert "R2" in metrics
    assert "Adjusted_R2" in metrics
    assert "MAPE" in metrics
    assert metrics["MAE"] == 10.0
    assert metrics["RMSE"] == 10.0
    assert metrics["R2"] > 0.9
