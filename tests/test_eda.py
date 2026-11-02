"""Unit tests for EDA and data audit functions."""

import pytest
import numpy as np
import pandas as pd

from src.data.audit import audit_dataframe, generate_audit_markdown
from src.analysis.eda import (
    class_balance_analysis,
    amount_distribution_analysis,
    fraud_rate_by_hour,
    correlation_with_class,
)
from src.utils.config import ALL_COLS


@pytest.fixture
def sample_eda_dataset():
    """Create reproducible data for testing EDA functions."""
    rng = np.random.default_rng(123)
    n = 200
    data = {f"V{i}": rng.normal(0, 1, n).astype(np.float32) for i in range(1, 29)}
    # Make V10 strongly negatively correlated with Class
    y = np.zeros(n, dtype=np.int8)
    y[:10] = 1  # 10 frauds
    data["Class"] = y
    data["V10"] = -3.0 * y + rng.normal(0, 0.5, n).astype(np.float32)

    # Time across 2 days (0 to 172800)
    data["Time"] = np.linspace(0, 172800, n, dtype=np.float32)
    # Amounts
    amounts = np.full(n, 50.0, dtype=np.float32)
    amounts[y == 1] = 200.0  # Frauds have higher amount
    data["Amount"] = amounts

    return pd.DataFrame(data)[ALL_COLS]


def test_class_balance_analysis(sample_eda_dataset):
    """Test class balance computation."""
    res = class_balance_analysis(sample_eda_dataset)
    assert res["total_transactions"] == 200
    assert res["fraud_count"] == 10
    assert res["normal_count"] == 190
    assert res["fraud_pct"] == 5.0
    assert res["normal_pct"] == 95.0


def test_amount_distribution_analysis(sample_eda_dataset):
    """Test amount distribution calculation."""
    res = amount_distribution_analysis(sample_eda_dataset)
    assert "normal" in res and "fraud" in res
    assert res["normal"]["mean"] == 50.0
    assert res["fraud"]["mean"] == 200.0


def test_fraud_rate_by_hour(sample_eda_dataset):
    """Test fraud rate grouped by 24 hours."""
    df_hourly = fraud_rate_by_hour(sample_eda_dataset)
    assert len(df_hourly) == 24
    assert "hour" in df_hourly.columns
    assert "fraud_rate" in df_hourly.columns
    assert df_hourly["total_transactions"].sum() == 200


def test_correlation_with_class(sample_eda_dataset):
    """Test Pearson correlation calculation."""
    corr_df = correlation_with_class(sample_eda_dataset)
    assert not corr_df.empty
    assert "feature" in corr_df.columns
    assert "correlation" in corr_df.columns

    # V10 was created to have strong negative correlation with Class
    top_feature = corr_df.iloc[0]["feature"]
    assert top_feature == "V10"
    assert corr_df.iloc[0]["correlation"] < -0.7


def test_audit_dataframe_and_markdown(sample_eda_dataset):
    """Test audit_dataframe and markdown generation."""
    audit_data = audit_dataframe(sample_eda_dataset, file_hash="abcdef123456")
    assert audit_data["shape"]["num_rows"] == 200
    assert audit_data["class_distribution"]["fraud_transactions"] == 10

    md = generate_audit_markdown(audit_data)
    assert "# 📋 Credit Card Fraud Dataset Audit Report" in md
    assert "abcdef123456" in md
