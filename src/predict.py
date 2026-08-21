"""Load the saved model and make predictions on new house data."""
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(__file__))
try:
    from src.utils import get_logger, load_object
except ImportError:
    from utils import get_logger, load_object

logger = get_logger(__name__)
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_artifacts():
    """Load the trained model, scaler, and feature column order."""
    model = load_object(os.path.join(MODELS_DIR, "best_model.pkl"))
    scaler = load_object(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_columns = load_object(os.path.join(MODELS_DIR, "feature_columns.pkl"))
    return model, scaler, feature_columns


def predict_price(input_dict: dict) -> float:
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
    df["rooms_per_household"] = df["total_rooms"] / df["households"].replace(0, 1)
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"].replace(0, 1)
    df["population_per_household"] = df["population"] / df["households"].replace(0, 1)
    df = pd.get_dummies(df, columns=["ocean_proximity"])
    df.columns = (
        df.columns.str.replace("<", "under_", regex=False)
        .str.replace(">", "over_", regex=False)
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
        .str.replace(" ", "_", regex=False)
    )

    # Align to the exact column set/order the model was trained on
    df = df.reindex(columns=feature_columns, fill_value=0)
    scaled = scaler.transform(df)
    prediction = model.predict(scaled)[0]
    return float(prediction)


if __name__ == "__main__":
    sample = {
        "longitude": -118.25, "latitude": 34.05, "housing_median_age": 25,
        "total_rooms": 3000, "total_bedrooms": 600, "population": 1400,
        "households": 550, "median_income": 5.5, "ocean_proximity": "NEAR OCEAN",
    }
    price = predict_price(sample)
    print(f"Predicted Median House Value: ${price:,.2f}")
