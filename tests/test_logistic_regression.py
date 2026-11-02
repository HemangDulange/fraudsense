"""Unit tests for LogisticRegression baseline from scratch."""

import pytest
import numpy as np

from src.models.logistic_regression import LogisticRegressionFromScratch


def test_logistic_regression_fit_and_probabilities():
    """Verify supervised logistic regression training, convergence, and scoring."""
    rng = np.random.default_rng(42)
    n_samples = 400
    n_features = 5

    # Linearly separable data
    X = rng.normal(0, 1, size=(n_samples, n_features)).astype(np.float32)
    # True hyperplane: positive if x0 + x1 > 0
    y = ((X[:, 0] + X[:, 1]) > 0.0).astype(np.float32)

    model = LogisticRegressionFromScratch(lr=0.1, n_epochs=300, l2_reg=0.01)
    model.fit(X, y)

    probs = model.anomaly_score(X)
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    preds = model.predict(X, threshold=0.5)
    accuracy = np.mean(preds == y)
    assert accuracy > 0.85


def test_logistic_regression_class_weighting():
    """Verify that class weighting enables learning under severe imbalance."""
    rng = np.random.default_rng(99)
    n_samples = 500
    X = rng.normal(0, 1, size=(n_samples, 4)).astype(np.float32)
    y = np.zeros(n_samples, dtype=np.float32)

    # Only 10 positive cases (2% positive), shifted along feature 0
    pos_idx = rng.choice(n_samples, size=10, replace=False)
    y[pos_idx] = 1.0
    X[pos_idx, 0] += 5.0

    model = LogisticRegressionFromScratch(lr=0.1, n_epochs=200, use_class_weights=True)
    model.fit(X, y)

    # Model should assign much higher probability to the 10 positive cases
    pos_scores = model.anomaly_score(X[pos_idx])
    neg_scores = model.anomaly_score(X[y == 0])

    assert np.mean(pos_scores) > np.mean(neg_scores) + 0.5


def test_logistic_regression_state_roundtrip():
    """Verify state serialization without pickle."""
    rng = np.random.default_rng(123)
    X = rng.normal(0, 1, size=(100, 3)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.float32)

    model = LogisticRegressionFromScratch(lr=0.05, n_epochs=100)
    model.fit(X, y)
    scores1 = model.anomaly_score(X[:5])

    state = model.to_state()
    reconstructed = LogisticRegressionFromScratch.from_state(state)
    scores2 = reconstructed.anomaly_score(X[:5])

    np.testing.assert_allclose(scores1, scores2, rtol=1e-5)
