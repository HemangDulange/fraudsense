"""Unit tests for Gaussian, Mahalanobis, and PCA anomaly detectors."""

import pytest
import numpy as np

from src.models.gaussian_detector import GaussianDetector
from src.models.mahalanobis_detector import MahalanobisDetector
from src.models.pca_detector import PCADetector


@pytest.fixture
def synthetic_data():
    """Create synthetic normal data and anomalous points."""
    rng = np.random.default_rng(42)
    n_samples = 500
    n_features = 10

    # Normal cluster centered at 0 with standard variance
    X_normal = rng.normal(0, 1, size=(n_samples, n_features)).astype(np.float32)

    # Extreme anomalies placed 10 standard deviations away
    X_anomalous = rng.normal(10, 1, size=(20, n_features)).astype(np.float32)

    return X_normal, X_anomalous


def test_gaussian_detector(synthetic_data):
    """Test GaussianDetector scoring, smoothing, and state serialization."""
    X_normal, X_anomalous = synthetic_data

    for smoothing in [1e-9, 1e-6, 1e-3]:
        model = GaussianDetector(var_smoothing=smoothing)
        model.fit(X_normal)

        normal_scores = model.anomaly_score(X_normal)
        anomaly_scores = model.anomaly_score(X_anomalous)

        # Anomalies must produce significantly higher negative log-likelihood scores
        assert np.mean(anomaly_scores) > np.mean(normal_scores) + 50.0

        # Verify serialization roundtrip
        state = model.to_state()
        reconstructed = GaussianDetector.from_state(state)
        new_scores = reconstructed.anomaly_score(X_anomalous)
        np.testing.assert_allclose(anomaly_scores, new_scores, rtol=1e-5)


def test_mahalanobis_detector(synthetic_data):
    """Test MahalanobisDetector with regularized covariance and np.linalg.solve."""
    X_normal, X_anomalous = synthetic_data

    for reg in [1e-6, 1e-4, 1e-2]:
        model = MahalanobisDetector(reg=reg)
        model.fit(X_normal)

        normal_scores = model.anomaly_score(X_normal)
        anomaly_scores = model.anomaly_score(X_anomalous)

        # Anomalous points should yield large squared Mahalanobis distances
        assert np.mean(anomaly_scores) > np.mean(normal_scores) + 50.0
        assert np.all(normal_scores >= 0.0)

        # Verify serialization roundtrip
        state = model.to_state()
        reconstructed = MahalanobisDetector.from_state(state)
        new_scores = reconstructed.anomaly_score(X_anomalous)
        np.testing.assert_allclose(anomaly_scores, new_scores, rtol=1e-5)


def test_mahalanobis_score_of_mean_vector_is_zero():
    """Verify that the Mahalanobis anomaly score of the exact mean vector is 0.0."""
    rng = np.random.default_rng(123)
    X = rng.normal(5.0, 2.0, size=(100, 8)).astype(np.float32)
    model = MahalanobisDetector(reg=1e-4)
    model.fit(X)

    # Score of the fitted mean vector
    mean_vec = model.mean_.reshape(1, -1)
    score = model.anomaly_score(mean_vec)
    assert pytest.approx(float(score[0]), abs=1e-6) == 0.0


def test_pca_detector(synthetic_data):
    """Test PCADetector reconstruction error and component options."""
    X_normal, X_anomalous = synthetic_data

    for n_comp in [5, 8]:  # n_features is 10 in synthetic_data fixture
        model = PCADetector(n_components=n_comp)
        model.fit(X_normal)

        normal_scores = model.anomaly_score(X_normal)
        anomaly_scores = model.anomaly_score(X_anomalous)

        # Anomalies should have substantially higher reconstruction error
        assert np.mean(anomaly_scores) > np.mean(normal_scores) + 20.0

        # Verify serialization roundtrip
        state = model.to_state()
        reconstructed = PCADetector.from_state(state)
        new_scores = reconstructed.anomaly_score(X_anomalous)
        np.testing.assert_allclose(anomaly_scores, new_scores, rtol=1e-5)
