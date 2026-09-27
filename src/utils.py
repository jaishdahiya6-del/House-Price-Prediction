"""Utility helpers: logging setup, seed setting, and generic save/load functions."""
import logging
import os
import random
from typing import Any
import numpy as np
import joblib


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger that prints to console with standard formatting.

    Args:
        name: Logger name.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across Python's random and numpy.

    Args:
        seed: Integer random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_object(obj: Any, path: str) -> None:
    """Save any Python object to disk using joblib.

    Args:
        obj: Object to serialize.
        path: Destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)


def load_object(path: str) -> Any:
    """Load a joblib-serialized object from disk.

    Args:
        path: File path of saved object.

    Returns:
        Deserialized Python object.
    """
    return joblib.load(path)
