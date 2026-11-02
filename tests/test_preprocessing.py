"""Unit tests for data preprocessing and stratified splitting."""

import pytest
import numpy as np
import pandas as pd

from src.data.preprocessing import stratified_split, Preprocessor
from src.utils.config import ALL_COLS, FEATURE_COLS


@pytest.fixture
def dummy_dataset():
    """Create a reproducible synthetic dataset for testing preprocessing."""
    rng = np.random.default_rng(42)
    n_samples = 1000
    n_frauds = 20  # 2% fraud rate

    data = {f"V{i}": rng.normal(0, 1, n_samples).astype(np.float32) for i in range(1, 29)}
    data["Time"] = rng.uniform(0, 172800, n_samples).astype(np.float32)
    data["Amount"] = rng.exponential(50, n_samples).astype(np.float32)

    # Insert labels: exactly n_frauds 1s and rest 0s
    classes = np.zeros(n_samples, dtype=np.int8)
    fraud_indices = rng.choice(n_samples, size=n_frauds, replace=False)
    classes[fraud_indices] = 1
    data["Class"] = classes

    return pd.DataFrame(data)[ALL_COLS]


def test_stratified_split_ratios_and_disjoint(dummy_dataset):
    """Test that stratified_split produces disjoint splits preserving class ratio."""
    y = dummy_dataset["Class"].to_numpy()
    train_idx, val_idx, test_idx = stratified_split(y, ratios=(0.6, 0.2, 0.2), seed=42)

    total_samples = len(y)
    assert len(train_idx) + len(val_idx) + len(test_idx) == total_samples

    # Verify all splits are strictly disjoint (no duplicate row indices)
    all_indices = np.concatenate([train_idx, val_idx, test_idx])
    assert len(np.unique(all_indices)) == total_samples

    # Verify fraud ratio is preserved across splits
    overall_fraud_rate = np.mean(y == 1)
    train_fraud_rate = np.mean(y[train_idx] == 1)
    val_fraud_rate = np.mean(y[val_idx] == 1)
    test_fraud_rate = np.mean(y[test_idx] == 1)

    assert pytest.approx(train_fraud_rate, abs=0.01) == overall_fraud_rate
    assert pytest.approx(val_fraud_rate, abs=0.01) == overall_fraud_rate
    assert pytest.approx(test_fraud_rate, abs=0.01) == overall_fraud_rate


def test_stratified_split_invalid_ratios():
    """Verify ValueError is raised when split ratios do not sum to 1."""
    y = np.array([0, 1, 0, 1, 0, 0])
    with pytest.raises(ValueError):
        stratified_split(y, ratios=(0.5, 0.2, 0.2))


def test_preprocessor_fit_transform_normal_only(dummy_dataset):
    """Verify that Preprocessor computes mean/std ONLY on normal (Class=0) rows."""
    prep = Preprocessor()
    
    # Add an extreme outlier in fraud rows
    df_with_outlier = dummy_dataset.copy()
    fraud_mask = df_with_outlier["Class"] == 1
    df_with_outlier.loc[fraud_mask, "V1"] = 10000.0

    # Fit preprocessor
    prep.fit(df_with_outlier)

    # The learned mean for V1 should be close to 0 (since normal rows have mean ~0),
    # proving the fraud outlier was completely excluded from fitting
    v1_idx = prep.feature_order.index("V1")
    learned_v1_mean = prep.mean_[v1_idx]
    assert abs(learned_v1_mean) < 0.2  # Should be close to 0, NOT pulled up by 10,000


def test_preprocessor_amount_and_time_transform():
    """Verify Amount is transformed by log1p and Time mapped to 0-24 hour range."""
    prep = Preprocessor()
    
    # 2 sample rows: Time = 7200 (2 hours), Amount = 0; Time = 3600 (1 hour), Amount = np.e - 1
    sample_df = pd.DataFrame({
        "Time": [7200.0, 3600.0],
        "Amount": [0.0, float(np.e - 1.0)],
        "Class": [0, 0],
    })
    for i in range(1, 29):
        sample_df[f"V{i}"] = [0.0, 0.0]

    transformed = prep._extract_and_transform_features(sample_df)
    time_idx = prep.feature_order.index("Time")
    amount_idx = prep.feature_order.index("Amount")

    # Time in hours
    assert pytest.approx(transformed[0, time_idx], rel=1e-4) == 2.0
    assert pytest.approx(transformed[1, time_idx], rel=1e-4) == 1.0

    # Amount in log1p: log1p(0) = 0, log1p(e - 1) = ln(e) = 1.0
    assert pytest.approx(transformed[0, amount_idx], abs=1e-4) == 0.0
    assert pytest.approx(transformed[1, amount_idx], abs=1e-4) == 1.0


def test_preprocessor_clipping(dummy_dataset):
    """Verify that extreme z-scores are clipped to [-10, 10]."""
    prep = Preprocessor(clip_range=(-10.0, 10.0))
    prep.fit(dummy_dataset)

    # Create synthetic test sample with huge positive and negative values
    extreme_df = dummy_dataset.head(2).copy()
    extreme_df.loc[0, "V2"] = 1e6
    extreme_df.loc[1, "V2"] = -1e6

    z = prep.transform(extreme_df)
    v2_idx = prep.feature_order.index("V2")

    assert z[0, v2_idx] == 10.0
    assert z[1, v2_idx] == -10.0


def test_preprocessor_state_serialization(dummy_dataset):
    """Verify to_state and from_state serialization without pickle."""
    prep1 = Preprocessor()
    prep1.fit(dummy_dataset)
    z1 = prep1.transform(dummy_dataset.head(5))

    # Serialize to dictionary state
    state = prep1.to_state()
    assert isinstance(state, dict)
    assert "mean" in state
    assert "std" in state

    # Reconstruct new preprocessor from state
    prep2 = Preprocessor.from_state(state)
    z2 = prep2.transform(dummy_dataset.head(5))

    # Results must match exactly
    np.testing.assert_allclose(z1, z2, rtol=1e-6)
