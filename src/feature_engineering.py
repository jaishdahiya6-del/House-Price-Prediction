"""Feature engineering: derived ratios, scaling, and feature/target prep."""
import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils import get_logger

logger = get_logger(__name__)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-informed ratio features for the California Housing data."""
    df = df.copy()
    df["rooms_per_household"] = df["total_rooms"] / df["households"].replace(0, 1)
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"].replace(0, 1)
    df["population_per_household"] = df["population"] / df["households"].replace(0, 1)
    logger.info("Added 3 derived features: rooms_per_household, "
                "bedrooms_per_room, population_per_household.")
    return df


def encode_categorical(df: pd.DataFrame, columns=("ocean_proximity",)) -> pd.DataFrame:
    """One-hot encode categorical columns and sanitize resulting column names
    (some downstream libraries like XGBoost reject '<', '>', '[', ']')."""
    df = pd.get_dummies(df, columns=list(columns), drop_first=False)
    df.columns = (
        df.columns.str.replace("<", "under_", regex=False)
        .str.replace(">", "over_", regex=False)
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
        .str.replace(" ", "_", regex=False)
    )
    logger.info(f"One-hot encoded categorical columns: {list(columns)}")
    return df


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Fit a StandardScaler on train data and transform both sets.

    Returns:
        (X_train_scaled_df, X_test_scaled_df, fitted_scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    logger.info("Scaled features using StandardScaler.")
    return X_train_scaled, X_test_scaled, scaler
