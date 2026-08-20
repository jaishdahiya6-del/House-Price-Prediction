"""Unit tests for prediction module."""
from src.predict import predict_price, load_artifacts


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
    assert price > 0
