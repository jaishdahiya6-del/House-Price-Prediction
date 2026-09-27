"""Unit tests for prediction module."""
import os
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from src.predict import predict_price, load_artifacts
from src.utils import save_object


@pytest.fixture(autouse=True)
def setup_dummy_artifacts(tmp_path, monkeypatch):
    """Ensure dummy model artifacts exist for tests if missing in models/."""
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    best_model_path = os.path.join(models_dir, "best_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    cols_path = os.path.join(models_dir, "feature_columns.pkl")

    if not (os.path.exists(best_model_path) and os.path.exists(scaler_path) and os.path.exists(cols_path)):
        # Create mock artifacts
        feature_columns = [
            "longitude", "latitude", "housing_median_age", "total_rooms",
            "total_bedrooms", "population", "households", "median_income",
            "rooms_per_household", "bedrooms_per_room", "population_per_household",
            "ocean_proximity_INLAND", "ocean_proximity_NEAR_OCEAN"
        ]
        X_dummy = pd.DataFrame(np.random.randn(10, len(feature_columns)), columns=feature_columns)
        y_dummy = np.random.randn(10) * 100000 + 200000

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_dummy)

        model = LinearRegression()
        model.fit(X_scaled, y_dummy)

        save_object(model, best_model_path)
        save_object(scaler, scaler_path)
        save_object(feature_columns, cols_path)


def test_load_artifacts():
    model, scaler, feature_columns = load_artifacts()
    assert model is not None
    assert scaler is not None
    assert len(feature_columns) > 0


def test_predict_price():
    sample = {
        "longitude": -118.25,
        "latitude": 34.05,
        "housing_median_age": 25,
        "total_rooms": 3000,
        "total_bedrooms": 600,
        "population": 1400,
        "households": 550,
        "median_income": 5.5,
        "ocean_proximity": "NEAR OCEAN",
    }
    price = predict_price(sample)
    assert isinstance(price, float)


def test_predict_price_missing_artifact(monkeypatch, tmp_path):
    """Test that predict.py raises FileNotFoundError when artifacts are missing."""
    fake_dir = str(tmp_path / "non_existent_models")
    monkeypatch.setattr("src.predict.MODELS_DIR", fake_dir)
    with pytest.raises(FileNotFoundError, match="Required model artifact missing"):
        load_artifacts()
