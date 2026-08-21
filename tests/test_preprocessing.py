"""Unit tests for preprocessing module."""
import numpy as np
import pandas as pd
from src.preprocessing import handle_missing_values, cap_outliers_iqr, split_data


def test_handle_missing_values():
    df = pd.DataFrame({
        "num1": [1.0, 2.0, np.nan, 4.0, 5.0],
        "cat1": ["a", "b", "c", "d", "e"]
    })
    cleaned_df = handle_missing_values(df)
    assert cleaned_df["num1"].isnull().sum() == 0
    assert cleaned_df["num1"].iloc[2] == 3.0  # median of [1,2,4,5] is 3.0


def test_cap_outliers_iqr():
    df = pd.DataFrame({
        "val": [10, 12, 11, 13, 12, 11, 10, 1000]  # 1000 is an outlier
    })
    capped_df = cap_outliers_iqr(df, columns=["val"], factor=1.5)
    assert capped_df["val"].max() < 1000


def test_split_data():
    df = pd.DataFrame({
        "f1": list(range(100)),
        "f2": list(range(100)),
        "target": list(range(100))
    })
    X_train, X_test, y_train, y_test = split_data(df, target="target", test_size=0.2, random_state=42)
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20
