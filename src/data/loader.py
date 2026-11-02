"""Data loading module for FraudSense.

This module provides functions to load the raw credit card transaction
dataset from disk, validate that all required columns are present, and
optimize memory usage with 32-bit floats.
"""

from pathlib import Path
from typing import Optional
import pandas as pd

from src.utils.config import (
    RAW_DATA_FILE,
    ALL_COLS,
    NUM_EXPECTED_COLS,
    DTYPE_MAPPING,
)


def load_raw(file_path: Optional[Path] = None) -> pd.DataFrame:
    """Load the raw Credit Card Fraud dataset into a pandas DataFrame.

    This function reads the CSV file directly with float32 types to save RAM,
    validates that all expected 31 columns exist, and provides a clear error
    message if the file is missing.

    Parameters
    ----------
    file_path : Optional[Path], default=None
        Path to the raw CSV file. If None, uses RAW_DATA_FILE
        (data/raw/creditcard.csv).

    Returns
    -------
    pd.DataFrame
        The loaded and validated credit card transaction dataset.

    Raises
    ------
    FileNotFoundError
        If the raw dataset file does not exist at the specified path.
    ValueError
        If the dataset columns do not match the expected 31 columns.
    """
    target_path = Path(file_path) if file_path is not None else RAW_DATA_FILE

    # 1. Check if the file exists. If not, guide the user on how to obtain it.
    if not target_path.exists():
        raise FileNotFoundError(
            f"\n[FraudSense Error] Raw dataset not found at:\n  --> {target_path.resolve()}\n\n"
            f"Please download 'creditcard.csv' from Kaggle:\n"
            f"  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n\n"
            f"After downloading, place the unzipped 'creditcard.csv' file inside:\n"
            f"  {target_path.parent.resolve()}/\n"
        )

    # 2. Read the CSV file.
    # We specify dtype=DTYPE_MAPPING so pandas loads numerical columns as float32
    # and the label as int8. This uses roughly half the memory of standard float64.
    df = pd.read_csv(target_path, dtype=DTYPE_MAPPING)

    # 3. Validate columns
    validate_columns(df)

    return df


def validate_columns(df: pd.DataFrame) -> None:
    """Validate that the DataFrame contains exactly the expected 31 columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check.

    Raises
    ------
    ValueError
        If any expected column is missing or if the column count is wrong.
    """
    actual_cols = list(df.columns)
    missing_cols = [col for col in ALL_COLS if col not in actual_cols]
    extra_cols = [col for col in actual_cols if col not in ALL_COLS]

    if missing_cols or extra_cols:
        error_msg = [
            f"[FraudSense Error] Invalid columns in dataset. Expected {NUM_EXPECTED_COLS} columns, but found {len(actual_cols)}."
        ]
        if missing_cols:
            error_msg.append(f"Missing columns: {missing_cols}")
        if extra_cols:
            error_msg.append(f"Unexpected extra columns: {extra_cols}")
        raise ValueError("\n".join(error_msg))
