"""Unit tests for Isolation Forest implemented from scratch with NumPy."""

import time
import pytest
import numpy as np

from src.models.isolation_forest import (
    IsolationForestFromScratch,
    c_factor,
)


def test_c_factor_math():
    """Verify c(n) BST average path length values."""
    assert c_factor(0) == 0.0
    assert c_factor(1) == 0.0
    assert c_factor(2) == 1.0
    assert c_factor(256) > 8.0  # ~9.0 for n=256


def test_isolation_forest_scoring_and_anomaly_separation():
    """Verify that IsolationForest scores outliers higher than normal points."""
    rng = np.random.default_rng(42)
    X_normal = rng.normal(0, 1, size=(500, 10)).astype(np.float32)
    X_anomalies = rng.uniform(8, 12, size=(20, 10)).astype(np.float32)

    model = IsolationForestFromScratch(n_trees=50, subsample_size=256, random_state=42)
    model.fit(X_normal)

    normal_scores = model.anomaly_score(X_normal)
    anomaly_scores = model.anomaly_score(X_anomalies)

    # All scores must be strictly between 0 and 1
    assert np.all((normal_scores > 0.0) & (normal_scores <= 1.0))
    assert np.all((anomaly_scores > 0.0) & (anomaly_scores <= 1.0))

    # Anomalies should produce significantly higher anomaly scores (shorter paths)
    assert np.mean(anomaly_scores) > np.mean(normal_scores) + 0.15


def test_isolation_forest_state_roundtrip():
    """Verify serialization without pickle."""
    rng = np.random.default_rng(101)
    X = rng.normal(0, 1, size=(200, 5)).astype(np.float32)

    model = IsolationForestFromScratch(n_trees=20, subsample_size=128, random_state=101)
    model.fit(X)
    scores1 = model.anomaly_score(X[:10])

    state = model.to_state()
    reconstructed = IsolationForestFromScratch.from_state(state)
    scores2 = reconstructed.anomaly_score(X[:10])

    np.testing.assert_allclose(scores1, scores2, rtol=1e-5)


def test_isolation_forest_vectorized_scoring_performance():
    """Performance requirement: scoring 50,000 rows must take under 30 seconds."""
    rng = np.random.default_rng(77)
    # 50,000 rows x 30 features
    X_normal = rng.normal(0, 1, size=(1000, 30)).astype(np.float32)
    X_benchmark = rng.normal(0, 1, size=(50000, 30)).astype(np.float32)

    model = IsolationForestFromScratch(n_trees=50, subsample_size=256, random_state=77)
    model.fit(X_normal)

    start_time = time.perf_counter()
    scores = model.anomaly_score(X_benchmark)
    elapsed = time.perf_counter() - start_time

    assert len(scores) == 50000
    # Must be under 30 seconds
    assert elapsed < 30.0, f"Scoring 50,000 rows took {elapsed:.2f}s (must be < 30s)"
