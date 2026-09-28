"""Loads the California Housing dataset (real-world, 20,640 rows, 1990 census).

Source: Pace, R. Kelley and Ronald Barry (1997), classic housing dataset,
mirrored on GitHub for reliable programmatic access.
"""
import os
import urllib.request
import pandas as pd

try:
    from src.utils import get_logger
except ImportError:
    from utils import get_logger

logger = get_logger(__name__)

DATA_URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "housing.csv")


def load_data(save_raw: bool = True) -> pd.DataFrame:
    """Fetch the California Housing dataset as a pandas DataFrame.

    Downloads from a GitHub-hosted CSV mirror on first run and caches
    locally at data/raw/housing.csv for subsequent runs.

    Args:
        save_raw: Whether to cache raw CSV locally if downloaded.

    Returns:
        DataFrame with 9 numeric/categorical features + target column
        'median_house_value'.
    """
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)

    if os.path.exists(RAW_DATA_PATH):
        logger.info(f"Loading cached dataset from {RAW_DATA_PATH}")
        df = pd.read_csv(RAW_DATA_PATH)
    else:
        logger.info(f"Downloading California Housing dataset from {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, RAW_DATA_PATH)
        df = pd.read_csv(RAW_DATA_PATH)
        if save_raw:
            logger.info(f"Raw data cached at {RAW_DATA_PATH}")

    logger.info(f"Loaded dataset with shape {df.shape}")
    return df


if __name__ == "__main__":
    data = load_data()
    logger.info(f"Dataset head:\n{data.head()}")
    logger.info(f"Dataset summary:\n{data.describe()}")
