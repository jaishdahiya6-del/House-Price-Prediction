"""Utility helpers: logging setup and generic save/load functions."""
import logging
import os
import joblib


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger that prints to console."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def save_object(obj, path: str) -> None:
    """Save any Python object to disk using joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)


def load_object(path: str):
    """Load a joblib-serialized object from disk."""
    return joblib.load(path)
