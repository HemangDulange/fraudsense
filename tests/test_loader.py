"""Tests for configuration and data loader."""

import pytest
import pandas as pd
from pathlib import Path
from src.utils.config import (
    PROJECT_ROOT,
    RAW_DATA_FILE,
    SAMPLE_DATA_FILE,
    RANDOM_SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    ALL_COLS,
    FEATURE_COLS,
    TARGET_COL,
    NUM_EXPECTED_COLS,
)
from src.data.loader import load_raw, validate_columns


def test_config_constants():
    """Verify that configuration constants match project specifications."""
    assert RANDOM_SEED == 42
    assert TRAIN_RATIO == 0.6
    assert VAL_RATIO == 0.2
    assert TEST_RATIO == 0.2
    assert len(ALL_COLS) == 31
    assert NUM_EXPECTED_COLS == 31
    assert len(FEATURE_COLS) == 30
    assert TARGET_COL == "Class"
    assert ALL_COLS[0] == "Time"
    assert ALL_COLS[-2] == "Amount"
    assert ALL_COLS[-1] == "Class"


def test_loader_missing_file_raises_clear_error(tmp_path):
    """Verify that missing raw data file raises FileNotFoundError with download instructions."""
    non_existent = tmp_path / "creditcard.csv"
    with pytest.raises(FileNotFoundError) as excinfo:
        load_raw(non_existent)

    error_text = str(excinfo.value)
    assert "creditcard.csv" in error_text
    assert "kaggle" in error_text.lower()


def test_validate_columns_success():
    """Verify validate_columns passes when all 31 columns are present."""
    data = {col: [0.0] for col in ALL_COLS}
    df = pd.DataFrame(data)
    validate_columns(df)  # Should not raise


def test_validate_columns_missing():
    """Verify validate_columns raises ValueError when columns are missing."""
    data = {col: [0.0] for col in ALL_COLS[:-1]}  # missing 'Class'
    df = pd.DataFrame(data)
    with pytest.raises(ValueError) as excinfo:
        validate_columns(df)
    assert "Class" in str(excinfo.value)


def test_load_sample_file():
    """Verify that load_raw can successfully read the sample dataset."""
    df = load_raw(SAMPLE_DATA_FILE)
    assert len(df.columns) == 31
    assert df["Time"].dtype == "float32"
    assert df["V1"].dtype == "float32"
    assert df["Class"].dtype == "int8"
