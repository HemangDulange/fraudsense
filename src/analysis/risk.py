"""Risk scoring and risk band classification for FraudSense.

Everyday Analogy:
-----------------
Think of a credit score or a fever thermometer. A raw temperature of 38.8°C or a raw
anomaly score of 42.7 is difficult for a human operator or cashier to interpret on its own.
Instead, we convert the raw score into an intuitive 0-100 'Risk Percentile' by comparing
it to thousands of legitimate, verified transactions:
- 0 to 89: Normal everyday behavior (Low Risk -> Approve).
- 90 to Threshold: Suspicious, but not conclusively fraudulent (Medium Risk -> Step-up OTP verification).
- At or above Threshold: Extremely rare behavior consistent with fraud (High Risk -> Hold & Manual Review).

Crucially, this system provides decision support, not an automated verdict.
"""

from typing import Dict, Any, Union, Optional
from pathlib import Path
import json
import numpy as np


class RiskScorer:
    """Calculates empirical percentile risk scores and operational action bands.

    Parameters
    ----------
    val_normal_scores : np.ndarray
        Array of anomaly scores from verified normal transactions in the validation set.
    threshold : float
        The decision threshold chosen for this model (e.g., maximizing validation F1).
    model_name : str, default="deployed"
        Name of the model associated with these scores.
    """

    DISCLAIMER = "Decision support, not a verdict"

    def __init__(
        self,
        val_normal_scores: np.ndarray,
        threshold: float,
        model_name: str = "deployed",
    ):
        scores_arr = np.asarray(val_normal_scores, dtype=np.float32).ravel()
        if len(scores_arr) == 0:
            raise ValueError("val_normal_scores cannot be empty.")

        self.sorted_scores = np.sort(scores_arr)
        self.threshold = float(threshold)
        self.model_name = str(model_name)
        self.n_samples = len(self.sorted_scores)

        # Precalculate the percentile corresponding to the decision threshold
        self.threshold_percentile = float(self.percentile_rank(self.threshold))

    def percentile_rank(
        self,
        raw_scores: Union[float, int, np.ndarray, list],
    ) -> Union[float, np.ndarray]:
        """Compute the empirical CDF percentile rank (0-100) of raw scores.

        Uses np.searchsorted against the sorted baseline of normal validation scores.

        Parameters
        ----------
        raw_scores : Union[float, np.ndarray]
            Raw model anomaly score(s).

        Returns
        -------
        Union[float, np.ndarray]
            Percentile rank from 0.0 to 100.0.
        """
        is_scalar = np.isscalar(raw_scores) or isinstance(raw_scores, (float, int))
        scores = np.asarray(raw_scores, dtype=np.float32)

        # Count how many normal transactions scored <= this score
        ranks = np.searchsorted(self.sorted_scores, scores, side="right")
        pct = (ranks / float(self.n_samples)) * 100.0
        pct_clipped = np.clip(pct, 0.0, 100.0)

        if is_scalar:
            return float(pct_clipped)
        return pct_clipped.astype(np.float32)

    def assess_risk(self, raw_score: float) -> Dict[str, Any]:
        """Evaluate a transaction score and determine risk band and recommended action.

        Bands:
        - Low: risk below 90 -> 'Approve'
        - Medium: risk 90 up to deployed threshold -> 'Ask for OTP verification'
        - High: at or above deployed threshold -> 'Hold and send to manual review'

        Parameters
        ----------
        raw_score : float
            Raw anomaly score from the detector.

        Returns
        -------
        Dict[str, Any]
            Detailed assessment dictionary.
        """
        s = float(raw_score)
        pct = float(self.percentile_rank(s))

        # Check threshold first to handle models where threshold percentile is <= 90
        if s >= self.threshold:
            band = "High"
            action = "Hold and send to manual review"
            badge_color = "red"
        elif pct >= 90.0:
            band = "Medium"
            action = "Ask for OTP verification"
            badge_color = "orange"
        else:
            band = "Low"
            action = "Approve"
            badge_color = "green"

        return {
            "model_name": self.model_name,
            "raw_score": round(s, 6),
            "risk_score": round(pct, 2),
            "risk_band": band,
            "action": action,
            "badge_color": badge_color,
            "deployed_threshold": round(self.threshold, 6),
            "deployed_threshold_percentile": round(self.threshold_percentile, 2),
            "disclaimer": self.DISCLAIMER,
        }

    def to_state(self) -> Dict[str, Any]:
        """Serialize empirical CDF state to dictionary."""
        # Quantile downsample for saving if sample size is very large (e.g. > 1000 points)
        if len(self.sorted_scores) > 2000:
            quantiles = np.linspace(0.0, 1.0, 2000)
            saved_cdf = np.quantile(self.sorted_scores, quantiles).tolist()
        else:
            saved_cdf = self.sorted_scores.tolist()

        return {
            "model_name": self.model_name,
            "threshold": self.threshold,
            "threshold_percentile": self.threshold_percentile,
            "n_samples": self.n_samples,
            "sorted_scores": saved_cdf,
            "disclaimer": self.DISCLAIMER,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "RiskScorer":
        """Reconstruct RiskScorer from saved state."""
        instance = cls.__new__(cls)
        instance.model_name = state.get("model_name", "deployed")
        instance.threshold = float(state["threshold"])
        instance.threshold_percentile = float(state.get("threshold_percentile", 95.0))
        instance.sorted_scores = np.array(state["sorted_scores"], dtype=np.float32)
        instance.n_samples = int(state.get("n_samples", len(instance.sorted_scores)))
        return instance
