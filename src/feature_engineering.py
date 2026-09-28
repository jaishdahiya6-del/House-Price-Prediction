"""Feature engineering: derived ratios, scaling, and feature/target prep."""
from typing import Tuple, Sequence
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from src.utils import get_logger
except ImportError:
    from utils import get_logger

logger = get_logger(__name__)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-informed ratio features for the California Housing data.

    Args:
        df: Input DataFrame containing total_rooms, total_bedrooms, population, households.

    Returns:
        DataFrame with added rooms_per_household, bedrooms_per_room, and population_per_household.
    """
    df = df.copy()
    if "total_rooms" in df.columns and "households" in df.columns:
        df["rooms_per_household"] = df["total_rooms"] / df["households"].replace(0, 1)
    if "total_bedrooms" in df.columns and "total_rooms" in df.columns:
        df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"].replace(0, 1)
    if "population" in df.columns and "households" in df.columns:
        df["population_per_household"] = df["population"] / df["households"].replace(0, 1)

    logger.info("Added derived features: rooms_per_household, bedrooms_per_room, population_per_household.")
    return df


def encode_categorical(df: pd.DataFrame, columns: Sequence[str] = ("ocean_proximity",)) -> pd.DataFrame:
    """One-hot encode categorical columns and sanitize resulting column names
    (some downstream libraries like XGBoost reject '<', '>', '[', ']').

    Args:
        df: Input DataFrame.
        columns: Categorical columns to encode.

    Returns:
        DataFrame with encoded categorical features and sanitized column names.
    """
    cols_to_encode = [c for c in columns if c in df.columns]
    if not cols_to_encode:
        return df

    df = pd.get_dummies(df, columns=cols_to_encode, drop_first=False)
    df.columns = (
        df.columns.str.replace("<", "under_", regex=False)
        .str.replace(">", "over_", regex=False)
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
        .str.replace(" ", "_", regex=False)
    )
    logger.info(f"One-hot encoded categorical columns: {cols_to_encode}")
    return df


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit a StandardScaler on train data and transform both sets.

    Args:
        X_train: Training features DataFrame.
        X_test: Test features DataFrame.

    Returns:
        Tuple of (X_train_scaled_df, X_test_scaled_df, fitted_scaler).
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
