"""Multivariate Mahalanobis Distance Anomaly Detector from scratch with NumPy.

Everyday Analogy:
-----------------
Consider human height and shoe size. In the general population, taller people usually
wear larger shoes—height and shoe size are strongly correlated.
If someone is 6 feet 2 inches tall with size 11 shoes, both values are above average,
but the *combination* is completely normal.
However, if someone is 4 feet 5 inches tall and wears size 15 shoes, each individual
measurement might exist in nature, but the *combination* is bizarre.
Standard Euclidean distance treats features as if they were completely independent.
Mahalanobis distance accounts for the correlation between features: it stretches and
rotates the coordinate space according to the covariance matrix, measuring how far a point
lies from the normal multi-dimensional cluster.

Algorithm Description:
----------------------
Given normal transactions, we compute:
1. Mean vector: mu = mean(X_normal)
2. Sample covariance: Sigma = (X_normal - mu)^T (X_normal - mu) / (N - 1)
3. Ridge regularization: Sigma_reg = Sigma + reg * I
   (Adds a tiny positive ridge along the diagonal to ensure the matrix is strictly
   positive-definite and numerically invertible).

The squared Mahalanobis distance for a point x is:
   D_M^2(x) = (x - mu)^T * Sigma_reg^(-1) * (x - mu)

Instead of explicitly computing matrix inversion Sigma_reg^(-1) (which can be numerically
unstable), we solve the linear system:
   Sigma_reg * z = (x - mu)  ==>  z = Sigma_reg^(-1) * (x - mu)
using np.linalg.solve, and then compute the dot product: (x - mu)^T * z.
"""

from typing import Dict, Any, Optional
import numpy as np


class MahalanobisDetector:
    """Multivariate Mahalanobis Distance Anomaly Detector.

    Parameters
    ----------
    reg : float, default=1e-4
        Ridge regularization parameter added to the covariance matrix diagonal.
        Ensures invertibility and numerical stability. Common values: 1e-6, 1e-4, 1e-2.
    """

    def __init__(self, reg: float = 1e-4):
        if reg <= 0:
            raise ValueError(f"reg must be positive, got {reg}")
        self.reg = float(reg)
        self.mean_: Optional[np.ndarray] = None
        self.cov_reg_: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(self, X_normal: np.ndarray) -> "MahalanobisDetector":
        """Fit mean vector and regularized covariance matrix on normal transactions.

        Parameters
        ----------
        X_normal : np.ndarray
            2D array of normal transactions, shape (n_samples, n_features).

        Returns
        -------
        self : MahalanobisDetector
            Fitted detector.
        """
        X = np.asarray(X_normal, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")

        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError(f"Need at least 2 samples to compute covariance, got {n_samples}")

        # Compute mean vector
        mean = np.mean(X, axis=0)

        # Center the data: (X - mu)
        X_centered = X - mean

        # Unbiased empirical covariance: Sigma = (X_c^T @ X_c) / (N - 1)
        cov = (X_centered.T @ X_centered) / (n_samples - 1)

        # Ridge regularization: Sigma_reg = Sigma + reg * I
        identity = np.eye(n_features, dtype=np.float64)
        cov_reg = cov + self.reg * identity

        self.mean_ = mean.astype(np.float32)
        self.cov_reg_ = cov_reg.astype(np.float32)
        self.is_fitted = True

        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Compute squared Mahalanobis distance using np.linalg.solve.

        Higher score = greater distance from normal multivariate distribution = more suspicious.

        Parameters
        ----------
        X : np.ndarray
            2D array of transactions to score, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            1D float32 array of squared Mahalanobis distances, shape (n_samples,).
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        # Centered transactions: (x - mu)
        diff = X_arr - self.mean_  # shape: (n_samples, n_features)

        # Solve Sigma_reg @ Z = diff.T for Z
        # diff.T has shape (n_features, n_samples)
        # Z has shape (n_features, n_samples), where column i is Sigma_reg^(-1) @ diff[i]
        # Using np.linalg.solve avoids explicit matrix inversion (np.linalg.inv)
        cov_f64 = self.cov_reg_.astype(np.float64)
        diff_f64_T = diff.T.astype(np.float64)

        Z = np.linalg.solve(cov_f64, diff_f64_T)

        # Squared Mahalanobis distance is the dot product of diff[i] and Z[:, i]
        # Vectorized across all samples: sum along axis 0 of (diff.T * Z)
        squared_distances = np.sum(diff_f64_T * Z, axis=0)

        # Ensure no negative numerical artifacts
        squared_distances = np.maximum(0.0, squared_distances)

        return squared_distances.astype(np.float32)

    def to_state(self) -> Dict[str, Any]:
        """Serialize detector state to a dictionary without pickle."""
        if not self.is_fitted:
            raise RuntimeError("Cannot serialize an unfitted MahalanobisDetector.")

        return {
            "model_type": "MahalanobisDetector",
            "reg": self.reg,
            "mean": self.mean_.tolist(),
            "cov_reg": self.cov_reg_.tolist(),
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "MahalanobisDetector":
        """Reconstruct MahalanobisDetector from serialized state dictionary."""
        instance = cls(reg=state.get("reg", 1e-4))
        instance.mean_ = np.array(state["mean"], dtype=np.float32)
        instance.cov_reg_ = np.array(state["cov_reg"], dtype=np.float32)
        instance.is_fitted = bool(state.get("is_fitted", True))
        return instance
