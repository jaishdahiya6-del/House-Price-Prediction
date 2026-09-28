"""Load the saved model and make predictions on new house data."""
import os
import sys
from typing import Dict, Any, Tuple
import pandas as pd

sys.path.append(os.path.dirname(__file__))
try:
    from src.utils import get_logger, load_object
    from src.feature_engineering import add_derived_features, encode_categorical
except ImportError:
    from utils import get_logger, load_object
    from feature_engineering import add_derived_features, encode_categorical

logger = get_logger(__name__)
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_artifacts() -> Tuple[Any, Any, list]:
    """Load the trained model, scaler, and feature column order.

    Returns:
        Tuple of (model, scaler, feature_columns).

    Raises:
        FileNotFoundError: If any model artifact is missing from models/.
    """
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    cols_path = os.path.join(MODELS_DIR, "feature_columns.pkl")

    for p in (model_path, scaler_path, cols_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Required model artifact missing at '{p}'. Please run 'python src/train.py' first."
            )

    model = load_object(model_path)
    scaler = load_object(scaler_path)
    feature_columns = load_object(cols_path)
    return model, scaler, feature_columns


def predict_price(input_dict: Dict[str, Any]) -> float:
    """Predict median house value in USD for a single input record.

    Args:
        input_dict: Raw feature values, e.g.
            {"longitude": -118.25, "latitude": 34.05, "housing_median_age": 25,
             "total_rooms": 3000, "total_bedrooms": 600, "population": 1400,
             "households": 550, "median_income": 5.5, "ocean_proximity": "NEAR OCEAN"}

    Returns:
        Predicted median house value in USD.
    """
    model, scaler, feature_columns = load_artifacts()

    df = pd.DataFrame([input_dict])
    df = add_derived_features(df)
    if "ocean_proximity" in df.columns:
        df = encode_categorical(df, columns=["ocean_proximity"])

    # Align to the exact column set/order the model was trained on
    df = df.reindex(columns=feature_columns, fill_value=0)
    scaled = pd.DataFrame(scaler.transform(df), columns=feature_columns, index=df.index)
    prediction = model.predict(scaled)[0]
    return float(prediction)


if __name__ == "__main__":
    sample = {
        "longitude": -118.25, "latitude": 34.05, "housing_median_age": 25,
        "total_rooms": 3000, "total_bedrooms": 600, "population": 1400,
        "households": 550, "median_income": 5.5, "ocean_proximity": "NEAR OCEAN",
    }
    try:
        price = predict_price(sample)
        logger.info(f"Predicted Median House Value: ${price:,.2f}")
    except FileNotFoundError as err:
        logger.error(err)
