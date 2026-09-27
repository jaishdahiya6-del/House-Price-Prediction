"""Missing-value handling, outlier capping, and train/test splitting."""
from typing import List, Tuple, Union
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from src.utils import get_logger
except ImportError:
    from utils import get_logger

logger = get_logger(__name__)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing numeric values with the column median.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with missing numeric values imputed.
    """
    df = df.copy()
    missing_before = df.isnull().sum().sum()
    if missing_before > 0:
        for col in df.select_dtypes(include="number").columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        logger.info(f"Imputed {missing_before} missing values with median.")
    else:
        logger.info("No missing values found.")
    return df


def cap_outliers_iqr(df: pd.DataFrame, columns: Union[List[str], Tuple[str, ...]], factor: float = 1.5) -> pd.DataFrame:
    """Cap outliers in given numeric columns using the IQR rule (winsorizing).

    Args:
        df: Input DataFrame.
        columns: List/Tuple of column names to cap outliers for.
        factor: IQR multiplier factor for threshold boundary calculation.

    Returns:
        DataFrame with capped numeric columns.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        n_capped = int(((df[col] < lower) | (df[col] > upper)).sum())
        df[col] = df[col].clip(lower, upper)
        if n_capped:
            logger.info(f"Capped {n_capped} outliers in '{col}'.")
    return df


def split_data(
    df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split a DataFrame into train/test feature and target sets.

    Args:
        df: Input DataFrame.
        target: Target column name.
        test_size: Proportion of test dataset split.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(f"Split data -> train: {X_train.shape}, test: {X_test.shape}")
    return X_train, X_test, y_train, y_test
