"""PCA Reconstruction Error Anomaly Detector from scratch with NumPy.

Everyday Analogy:
-----------------
Imagine an art restorer who studies hundreds of authentic classical vases.
Although each vase is unique, they all share common shapes, curvature, and proportions.
If you take a 2D photograph of an authentic vase from a specific angle, the restorer
can reconstruct its full 3D shape with almost zero error because it follows the rules
of that pottery style.
Now imagine someone brings in a bizarre, mutated modern vase with jagged metal spikes.
When you take a 2D photograph and try to reconstruct the vase using the classical rules,
the spikes are completely missing in the reconstruction.
The difference (reconstruction error) between the actual object and its reconstructed
version will be massive. That reconstruction error immediately exposes it as an anomaly!

Algorithm Description:
----------------------
1. Fit Phase:
   - Compute mean vector: mu = mean(X_normal)
   - Compute sample covariance: C = (X_c^T @ X_c) / (N - 1)
   - Perform eigendecomposition using np.linalg.eigh(C) to find principal directions.
   - Sort eigenvectors by descending eigenvalues and keep the top k components (V_k).
   These k components capture the primary subspace where normal transactions lie.

2. Scoring Phase:
   - Center new transactions: X_c = X - mu
   - Project onto top k components: Z = X_c @ V_k  (compression)
   - Reconstruct back to full space: X_hat = Z @ V_k^T (decompression)
   - Compute reconstruction residual: R = X_c - X_hat
   - Anomaly score = squared Euclidean reconstruction error: ||R||_2^2 = sum_j (R_ij^2).
   Normal transactions lie near the principal subspace and reconstruct with tiny error.
   Fraudulent transactions violate normal correlations and produce huge reconstruction errors.
"""

from typing import Dict, Any, Optional
import numpy as np


class PCADetector:
    """PCA Reconstruction Error Anomaly Detector.

    Parameters
    ----------
    n_components : int, default=10
        Number of principal components to retain. Common values: 5, 10, 15, 20.
    """

    def __init__(self, n_components: int = 10):
        if n_components <= 0:
            raise ValueError(f"n_components must be positive, got {n_components}")
        self.n_components = int(n_components)
        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None  # shape: (n_features, n_components)
        self.explained_variance_: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(self, X_normal: np.ndarray) -> "PCADetector":
        """Compute principal components using np.linalg.eigh on covariance matrix of normal data.

        Parameters
        ----------
        X_normal : np.ndarray
            2D array of normal transactions, shape (n_samples, n_features).

        Returns
        -------
        self : PCADetector
            Fitted detector.
        """
        X = np.asarray(X_normal, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")

        n_samples, n_features = X.shape
        if self.n_components > n_features:
            raise ValueError(
                f"n_components ({self.n_components}) cannot exceed feature count ({n_features})"
            )

        # Center the data
        mean = np.mean(X, axis=0)
        X_centered = X - mean

        # Sample covariance matrix
        cov = (X_centered.T @ X_centered) / (n_samples - 1)

        # np.linalg.eigh computes eigenvalues and eigenvectors for symmetric matrices.
        # Eigenvalues are returned in ascending order.
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort in descending order (highest variance first)
        sort_idx = np.argsort(eigenvalues)[::-1]
        sorted_evals = eigenvalues[sort_idx]
        sorted_evecs = eigenvectors[:, sort_idx]

        # Retain top k principal components
        top_components = sorted_evecs[:, : self.n_components]
        top_variance = sorted_evals[: self.n_components]

        self.mean_ = mean.astype(np.float32)
        self.components_ = top_components.astype(np.float32)
        self.explained_variance_ = top_variance.astype(np.float32)
        self.is_fitted = True

        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Compute reconstruction error for each transaction.

        Higher score = poorer reconstruction by normal principal components = more suspicious.

        Parameters
        ----------
        X : np.ndarray
            2D array of transactions to score, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            1D float32 array of squared reconstruction errors, shape (n_samples,).
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        # Center the data
        X_centered = X_arr - self.mean_

        # Project into lower-dimensional subspace: Z = X_c @ V_k  (n_samples, n_components)
        Z = X_centered @ self.components_

        # Reconstruct back into original feature space: X_hat = Z @ V_k^T  (n_samples, n_features)
        X_reconstructed = Z @ self.components_.T

        # Reconstruction residual: R = X_c - X_hat
        residual = X_centered - X_reconstructed

        # Squared Euclidean reconstruction error: ||R||_2^2
        error = np.sum(residual**2, axis=1)

        return error.astype(np.float32)

    def to_state(self) -> Dict[str, Any]:
        """Serialize detector state to a dictionary without pickle."""
        if not self.is_fitted:
            raise RuntimeError("Cannot serialize an unfitted PCADetector.")

        return {
            "model_type": "PCADetector",
            "n_components": self.n_components,
            "mean": self.mean_.tolist(),
            "components": self.components_.tolist(),
            "explained_variance": self.explained_variance_.tolist(),
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "PCADetector":
        """Reconstruct PCADetector from serialized state dictionary."""
        instance = cls(n_components=state.get("n_components", 10))
        instance.mean_ = np.array(state["mean"], dtype=np.float32)
        instance.components_ = np.array(state["components"], dtype=np.float32)
        instance.explained_variance_ = np.array(state["explained_variance"], dtype=np.float32)
        instance.is_fitted = bool(state.get("is_fitted", True))
        return instance
