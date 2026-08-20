"""Unit tests for feature engineering module."""
import pandas as pd
import numpy as np
from src.feature_engineering import add_derived_features, encode_categorical, scale_features


def test_add_derived_features():
    df = pd.DataFrame({
        "total_rooms": [100, 200],
        "total_bedrooms": [20, 50],
        "population": [300, 400],
        "households": [50, 100]
    })
    engineered = add_derived_features(df)
    assert "rooms_per_household" in engineered.columns
    assert "bedrooms_per_room" in engineered.columns
    assert "population_per_household" in engineered.columns
    assert list(engineered["rooms_per_household"]) == [2.0, 2.0]


def test_encode_categorical():
    df = pd.DataFrame({
        "ocean_proximity": ["<1H OCEAN", "INLAND", "NEAR OCEAN"]
    })
    encoded = encode_categorical(df, columns=["ocean_proximity"])
    assert "ocean_proximity_under_1H_OCEAN" in encoded.columns
    assert "ocean_proximity_INLAND" in encoded.columns
    assert "ocean_proximity_NEAR_OCEAN" in encoded.columns


def test_scale_features():
    X_train = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0, 5.0]})
    X_test = pd.DataFrame({"f1": [2.0, 4.0]})

    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)
    assert np.isclose(X_train_s["f1"].mean(), 0.0, atol=1e-5)
    assert hasattr(scaler, "mean_")
