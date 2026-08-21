"""Missing-value handling, outlier capping, and train/test splitting."""
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from src.utils import get_logger
except ImportError:
    from utils import get_logger

logger = get_logger(__name__)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing numeric values with the column median."""
    missing_before = df.isnull().sum().sum()
    if missing_before > 0:
        for col in df.select_dtypes(include="number").columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        logger.info(f"Imputed {missing_before} missing values with median.")
    else:
        logger.info("No missing values found.")
    return df


def cap_outliers_iqr(df: pd.DataFrame, columns, factor: float = 1.5) -> pd.DataFrame:
    """Cap outliers in given numeric columns using the IQR rule (winsorizing)."""
    df = df.copy()
    for col in columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        n_capped = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower, upper)
        if n_capped:
            logger.info(f"Capped {n_capped} outliers in '{col}'.")
    return df


def split_data(
    df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split a DataFrame into train/test feature and target sets."""
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(f"Split data -> train: {X_train.shape}, test: {X_test.shape}")
    return X_train, X_test, y_train, y_test
