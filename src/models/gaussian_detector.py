"""Univariate Gaussian Anomaly Detector implemented from scratch with NumPy.

Everyday Analogy:
-----------------
Imagine an airport health-screening checkpoint. Every passenger's body temperature,
heart rate, and luggage weight are measured. In a healthy population, each measurement
follows a standard 'bell curve' (normal distribution). If a passenger's body temperature
is 104°F (40°C), that single feature is far out in the thin tail of the bell curve.
The detector calculates the probability of seeing each value; if the overall probability
is minuscule, it flags the passenger as an anomaly.

Algorithm Description:
----------------------
We assume each feature follows an independent 1D Gaussian distribution:
    p(x_j) = (1 / sqrt(2 * pi * sigma_j^2)) * exp(-(x_j - mu_j)^2 / (2 * sigma_j^2))

To avoid numerical underflow from multiplying small probabilities across many dimensions,
we take the negative log-likelihood (NLL) and sum across all features:
    -ln p(x) = sum_j [ 0.5 * ln(2 * pi) + 0.5 * ln(sigma_j^2) + (x_j - mu_j)^2 / (2 * sigma_j^2) ]

Transactions with higher NLL reside in low-density regions and are flagged as suspicious.
"""

from typing import Dict, Any, Optional
import numpy as np


class GaussianDetector:
    """Univariate Gaussian Anomaly Detector.

    Parameters
    ----------
    var_smoothing : float, default=1e-9
        Small constant added to variance for numerical stability, preventing
        division by zero if a feature has near-zero variance.
        Common values: 1e-9, 1e-6, 1e-3.
    """

    def __init__(self, var_smoothing: float = 1e-9):
        if var_smoothing <= 0:
            raise ValueError(f"var_smoothing must be positive, got {var_smoothing}")
        self.var_smoothing = float(var_smoothing)
        self.mean_: Optional[np.ndarray] = None
        self.var_: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(self, X_normal: np.ndarray) -> "GaussianDetector":
        """Fit independent Gaussian distributions to normal transactions.

        Parameters
        ----------
        X_normal : np.ndarray
            2D array of normal transactions of shape (n_samples, n_features).

        Returns
        -------
        self : GaussianDetector
            Fitted detector.
        """
        X = np.asarray(X_normal, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")

        # Compute empirical mean and variance along each feature column
        # Use float64 for intermediate summation to prevent numerical drift
        mean = np.mean(X, axis=0, dtype=np.float64)
        var = np.var(X, axis=0, dtype=np.float64) + self.var_smoothing

        self.mean_ = mean.astype(np.float32)
        self.var_ = var.astype(np.float32)
        self.is_fitted = True

        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly score as negative log-likelihood (NLL).

        Higher score = lower probability under normal Gaussian distribution = more suspicious.

        Parameters
        ----------
        X : np.ndarray
            2D array of transactions to score, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            1D float32 array of anomaly scores, shape (n_samples,).
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        # Difference from normal baseline mean: (x - mu)
        diff = X_arr - self.mean_

        # Negative log-likelihood calculation:
        # -ln p(x_j) = 0.5 * ln(2 * pi) + 0.5 * ln(var_j) + 0.5 * (x_j - mu_j)^2 / var_j
        # Summed over all features j
        log_2pi = np.log(2.0 * np.pi)
        n_features = X_arr.shape[1]

        const_term = 0.5 * n_features * log_2pi
        variance_term = 0.5 * np.sum(np.log(self.var_))
        distance_term = 0.5 * np.sum((diff**2) / self.var_, axis=1)

        nll_scores = const_term + variance_term + distance_term
        return nll_scores.astype(np.float32)

    def to_state(self) -> Dict[str, Any]:
        """Serialize detector state to a dictionary without pickle."""
        if not self.is_fitted:
            raise RuntimeError("Cannot serialize an unfitted GaussianDetector.")

        return {
            "model_type": "GaussianDetector",
            "var_smoothing": self.var_smoothing,
            "mean": self.mean_.tolist(),
            "var": self.var_.tolist(),
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "GaussianDetector":
        """Reconstruct GaussianDetector from serialized state dictionary."""
        instance = cls(var_smoothing=state.get("var_smoothing", 1e-9))
        instance.mean_ = np.array(state["mean"], dtype=np.float32)
        instance.var_ = np.array(state["var"], dtype=np.float32)
        instance.is_fitted = bool(state.get("is_fitted", True))
        return instance
