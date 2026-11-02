"""Data preprocessing and feature scaling for FraudSense.

IMPORTANT MACHINE LEARNING PRINCIPLE:
====================================
Why do we SPLIT FIRST and fit preprocessing ONLY on the training set?
---------------------------------------------------------------------
1. Prevent Data Leakage:
   In real-world fraud detection, future incoming transactions are completely unknown.
   If we computed the feature statistics (such as mean, standard deviation, or log scales)
   across the entire dataset before splitting, the test and validation sets would 'leak'
   distributional knowledge into the training process. This leads to overly optimistic
   evaluation metrics that inevitably fail in production.

2. Pure Anomaly Baseline:
   In anomaly detection, our goal is to model what 'normal' transactions look like.
   Therefore, the normalization parameters (mean and standard deviation) are learned
   STRICTLY from the NORMAL (Class = 0) rows in the TRAINING set. This ensures that:
   - Normal transactions have a mean of 0 and variance of 1.
   - Fraudulent transactions stand out as large statistical deviations (high z-scores).
"""

from typing import Tuple, Dict, Any, Optional, Union, List
import numpy as np
import pandas as pd

from src.utils.config import (
    FEATURE_COLS,
    TIME_COL,
    AMOUNT_COL,
    TARGET_COL,
)


def stratified_split(
    y: Union[np.ndarray, pd.Series],
    ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split indices into stratified train, validation, and test subsets.

    This function is implemented purely with NumPy and maintains the exact
    ratio of normal (0) to fraudulent (1) transactions across all three splits.

    Parameters
    ----------
    y : Union[np.ndarray, pd.Series]
        1D array or Series of binary class labels (0 = Normal, 1 = Fraud).
    ratios : Tuple[float, float, float], default=(0.6, 0.2, 0.2)
        Proportions for (train, validation, test). Must sum to 1.0.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    train_idx : np.ndarray
        1D array of integer indices for the training set.
    val_idx : np.ndarray
        1D array of integer indices for the validation set.
    test_idx : np.ndarray
        1D array of integer indices for the test set.
    """
    train_ratio, val_ratio, test_ratio = ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {train_ratio} + {val_ratio} + {test_ratio} = {sum(ratios)}"
        )

    # Convert y to 1D numpy array
    y_arr = np.asarray(y).ravel()

    # Create random number generator with fixed seed
    rng = np.random.default_rng(seed)

    # Separate normal (0) and fraud (1) indices
    idx_normal = np.where(y_arr == 0)[0]
    idx_fraud = np.where(y_arr == 1)[0]

    # Shuffle each class independently
    idx_normal_shuffled = rng.permutation(idx_normal)
    idx_fraud_shuffled = rng.permutation(idx_fraud)

    # Helper function to split an index array by the given ratios
    def _split_indices(indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(indices)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        train_part = indices[:n_train]
        val_part = indices[n_train : n_train + n_val]
        test_part = indices[n_train + n_val :]
        return train_part, val_part, test_part

    train_norm, val_norm, test_norm = _split_indices(idx_normal_shuffled)
    train_fraud, val_fraud, test_fraud = _split_indices(idx_fraud_shuffled)

    # Combine normal and fraud indices for each split and shuffle them
    train_idx = rng.permutation(np.concatenate([train_norm, train_fraud]))
    val_idx = rng.permutation(np.concatenate([val_norm, val_fraud]))
    test_idx = rng.permutation(np.concatenate([test_norm, test_fraud]))

    return train_idx, val_idx, test_idx


class Preprocessor:
    """Feature preprocessor for credit card fraud anomaly detection.

    Steps applied:
    1. Feature ordering: Enforces fixed column order [Time, V1..V28, Amount].
    2. Amount transformation: log1p(Amount) = ln(1 + Amount) to compress extreme dollar values.
    3. Time transformation: hour of day = (Time % 86400) / 3600 (maps seconds to a 0-24 hour scale).
    4. Standardization: (x - mean) / std learned ONLY from normal (Class=0) training rows.
    5. Numerical safety: Zero or near-zero std is replaced by 1.0 to prevent division by zero.
    6. Z-score clipping: Values are clipped to [-10, 10] to prevent extreme outliers from destabilizing numerical linear algebra.
    """

    def __init__(
        self,
        feature_order: Optional[List[str]] = None,
        clip_range: Tuple[float, float] = (-10.0, 10.0),
    ):
        self.feature_order = feature_order if feature_order is not None else list(FEATURE_COLS)
        self.clip_range = clip_range
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def _extract_and_transform_features(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """Extract ordered columns and apply Amount log1p and Time hour-of-day transforms.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Raw inputs containing Time, V1..V28, and Amount.

        Returns
        -------
        np.ndarray
            2D float32 array with transformed Time and Amount.
        """
        if isinstance(X, pd.DataFrame):
            # Check for missing feature columns
            missing = [c for c in self.feature_order if c not in X.columns]
            if missing:
                raise ValueError(f"Missing required feature columns: {missing}")
            # Extract features in fixed order
            data = X[self.feature_order].to_numpy(dtype=np.float32, copy=True)
        else:
            # NumPy array: must match length of feature_order
            data = np.asarray(X, dtype=np.float32).copy()
            if data.ndim == 1:
                data = data.reshape(1, -1)
            if data.shape[1] != len(self.feature_order):
                raise ValueError(
                    f"Expected input with {len(self.feature_order)} columns, got {data.shape[1]}"
                )

        # Locate Time and Amount column indices
        time_idx = self.feature_order.index(TIME_COL)
        amount_idx = self.feature_order.index(AMOUNT_COL)

        # 1. Transform Time -> hour of day: (Time % 86400) / 3600
        # 86,400 seconds = 24 hours in a day
        data[:, time_idx] = (data[:, time_idx] % 86400.0) / 3600.0

        # 2. Transform Amount -> log1p(Amount)
        # Using np.maximum(0.0, ...) protects against negative values if any exist
        data[:, amount_idx] = np.log1p(np.maximum(0.0, data[:, amount_idx]))

        return data

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> "Preprocessor":
        """Fit preprocessor by computing mean and std ONLY from normal (Class=0) rows.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Training data.
        y : Optional[Union[pd.Series, np.ndarray]], default=None
            Class labels (0 = Normal, 1 = Fraud). If None and X is a DataFrame
            containing a 'Class' column, 'Class' will be used to isolate normal rows.
            If y is not provided at all, all rows in X are assumed to be normal.

        Returns
        -------
        self : Preprocessor
            The fitted preprocessor instance.
        """
        # Determine labels if X is a DataFrame with 'Class' column
        labels = None
        if y is not None:
            labels = np.asarray(y).ravel()
        elif isinstance(X, pd.DataFrame) and TARGET_COL in X.columns:
            labels = X[TARGET_COL].to_numpy().ravel()

        # Transform raw features (Time -> hour, Amount -> log1p)
        transformed_data = self._extract_and_transform_features(X)

        # Filter strictly to normal transactions (Class = 0)
        if labels is not None:
            normal_mask = labels == 0
            if not np.any(normal_mask):
                raise ValueError("No normal (Class = 0) instances found to fit preprocessor.")
            fit_data = transformed_data[normal_mask]
        else:
            fit_data = transformed_data

        # Compute mean and standard deviation along each feature column
        # Using float64 for intermediate calculation to avoid accumulation error,
        # then casting back to float32.
        mean = np.mean(fit_data, axis=0, dtype=np.float64)
        std = np.std(fit_data, axis=0, dtype=np.float64)

        # Handle zero or near-zero standard deviation (replace with 1.0 to avoid division by zero)
        std = np.where(std < 1e-7, 1.0, std)

        self.mean_ = mean.astype(np.float32)
        self.std_ = std.astype(np.float32)
        self.is_fitted = True

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Standardize features and clip z-scores to [-10, 10].

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Features to transform.

        Returns
        -------
        np.ndarray
            Standardized, clipped 2D float32 array.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor has not been fitted yet. Call fit() before transform().")

        # Apply Time -> hour and Amount -> log1p transformations
        data = self._extract_and_transform_features(X)

        # Standardize using learned normal baseline parameters
        z = (data - self.mean_) / self.std_

        # Clip z-scores to [-10, 10]
        z_clipped = np.clip(z, self.clip_range[0], self.clip_range[1])

        return z_clipped.astype(np.float32)

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> np.ndarray:
        """Fit preprocessor on normal training rows and transform X."""
        return self.fit(X, y).transform(X)

    def to_state(self) -> Dict[str, Any]:
        """Serialize preprocessor state to a dictionary without pickle.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing feature order, mean, std, and clip boundaries.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot serialize an unfitted Preprocessor.")

        return {
            "feature_order": list(self.feature_order),
            "clip_range": [float(self.clip_range[0]), float(self.clip_range[1])],
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
            "is_fitted": bool(self.is_fitted),
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "Preprocessor":
        """Reconstruct a Preprocessor from a serialized state dictionary.

        Parameters
        ----------
        state : Dict[str, Any]
            State dictionary generated by to_state().

        Returns
        -------
        Preprocessor
            Reconstructed and ready-to-use Preprocessor instance.
        """
        instance = cls(
            feature_order=state["feature_order"],
            clip_range=tuple(state.get("clip_range", (-10.0, 10.0))),
        )
        instance.mean_ = np.array(state["mean"], dtype=np.float32)
        instance.std_ = np.array(state["std"], dtype=np.float32)
        instance.is_fitted = bool(state.get("is_fitted", True))
        return instance
