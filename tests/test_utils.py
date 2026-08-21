"""Unit tests for utility functions."""
import os
import random
import numpy as np
from src.utils import set_seed, save_object, load_object, get_logger


def test_set_seed():
    set_seed(42)
    r1 = random.random()
    n1 = np.random.rand()

    set_seed(42)
    r2 = random.random()
    n2 = np.random.rand()

    assert r1 == r2
    assert n1 == n2


def test_save_load_object(tmp_path):
    filepath = os.path.join(tmp_path, "test_obj.pkl")
    data = {"key": "value", "numbers": [1, 2, 3]}

    save_object(data, filepath)
    assert os.path.exists(filepath)

    loaded = load_object(filepath)
    assert loaded == data


def test_get_logger():
    logger = get_logger("test_logger")
    assert logger.name == "test_logger"
