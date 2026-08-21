"""Unit tests for data loader."""
import os
import pandas as pd
from src.data_loader import load_data, RAW_DATA_PATH


def test_load_data():
    df = load_data(save_raw=True)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "median_house_value" in df.columns
    assert os.path.exists(RAW_DATA_PATH)
