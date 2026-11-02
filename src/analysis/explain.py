"""Transaction-level anomaly explanation based on normal feature profiles.

Works universally across all models (Gaussian, Mahalanobis, PCA, Isolation Forest,
and Logistic Regression) because reasons are derived by comparing the transaction's
feature values against the baseline distribution learned from normal transactions.
"""

from typing import Dict, Any, List, Union, Optional
from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.utils.config import (
    FEATURE_COLS,
    TIME_COL,
    AMOUNT_COL,
    MODELS_DIR,
)
from src.data.preprocessing import Preprocessor


PRIVACY_NOTE = (
    "Note on interpretability: Features V1 through V28 are anonymized principal components "
    "from PCA to preserve cardholder privacy. Only 'Amount' (transaction amount in dollars) "
    "and 'Time' (hour of the day) carry direct real-world physical meaning."
)


def load_default_preprocessor() -> Preprocessor:
    """Load the preprocessor state saved in artifacts/models/preprocessor.json."""
    prep_path = MODELS_DIR / "preprocessor.json"
    if not prep_path.exists():
        raise FileNotFoundError(
            f"Preprocessor state not found at {prep_path}. Run scripts/train_models.py first."
        )
    with open(prep_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    return Preprocessor.from_state(state)


def explain(
    row: Union[pd.Series, pd.DataFrame, Dict[str, float], np.ndarray],
    preprocessor: Optional[Preprocessor] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Identify the top k feature deviations for a single transaction.

    Parameters
    ----------
    row : Union[pd.Series, pd.DataFrame, Dict[str, float], np.ndarray]
        A single transaction containing all 30 features.
    preprocessor : Optional[Preprocessor], default=None
        Fitted Preprocessor containing normal baseline mean and std.
        If None, automatically loads from artifacts/models/preprocessor.json.
    top_k : int, default=5
        Number of top anomalous feature contributions to return.

    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - top_reasons: list of top k feature deviation dicts
        - total_deviation: sum of absolute z-scores across all features
        - note: privacy disclaimer regarding V1-V28
    """
    if preprocessor is None:
        preprocessor = load_default_preprocessor()

    feature_order = preprocessor.feature_order

    # Convert row input to dictionary
    if isinstance(row, pd.Series):
        row_dict = row.to_dict()
    elif isinstance(row, pd.DataFrame):
        row_dict = row.iloc[0].to_dict()
    elif isinstance(row, dict):
        row_dict = row
    elif isinstance(row, np.ndarray):
        arr = row.ravel()
        if len(arr) != len(feature_order):
            raise ValueError(f"Array length {len(arr)} does not match {len(feature_order)} features.")
        row_dict = {col: float(arr[i]) for i, col in enumerate(feature_order)}
    else:
        raise TypeError(f"Unsupported row type: {type(row)}")

    # Extract raw values in fixed feature order
    raw_vals = np.array([float(row_dict[col]) for col in feature_order], dtype=np.float32)

    # Transform Time (hour) and Amount (log1p)
    time_idx = feature_order.index(TIME_COL)
    amount_idx = feature_order.index(AMOUNT_COL)

    transformed_vals = raw_vals.copy()
    raw_time_sec = float(raw_vals[time_idx])
    hour_of_day = float((raw_time_sec % 86400.0) / 3600.0)
    transformed_vals[time_idx] = hour_of_day

    raw_amount = float(raw_vals[amount_idx])
    transformed_vals[amount_idx] = float(np.log1p(max(0.0, raw_amount)))

    # Compute z-scores against normal baseline
    means = preprocessor.mean_
    stds = preprocessor.std_
    z_scores = (transformed_vals - means) / stds

    abs_z = np.abs(z_scores)
    total_deviation = float(np.sum(abs_z))
    if total_deviation < 1e-9:
        total_deviation = 1.0

    # Sort feature indices by largest absolute z-score
    ranked_indices = np.argsort(abs_z)[::-1][:top_k]

    reasons: List[Dict[str, Any]] = []
    for idx in ranked_indices:
        feat_name = feature_order[idx]
        z_val = float(z_scores[idx])
        raw_val = float(raw_vals[idx])
        transformed_val = float(transformed_vals[idx])
        norm_mean = float(means[idx])
        norm_std = float(stds[idx])
        share_pct = float((abs(z_val) / total_deviation) * 100.0)

        direction = "unusually high" if z_val > 0 else "unusually low"

        # Generate plain-English explanation sentence
        if feat_name == AMOUNT_COL:
            display_val = f"${raw_amount:.2f}"
            normal_display = f"${(np.expm1(norm_mean)):.2f}"
            sentence = (
                f"Transaction amount of {display_val} is {direction} compared to typical normal spending "
                f"(baseline ~{normal_display}, z-score: {z_val:+.2f}, accounting for {share_pct:.1f}% of total deviation)."
            )
        elif feat_name == TIME_COL:
            int_hour = int(hour_of_day)
            minute = int((hour_of_day - int_hour) * 60)
            ampm = "AM" if int_hour < 12 else "PM"
            clock_hour = int_hour if int_hour <= 12 else int_hour - 12
            if clock_hour == 0:
                clock_hour = 12
            time_str = f"{clock_hour:02d}:{minute:02d} {ampm}"
            display_val = f"{hour_of_day:.1f}h ({time_str})"
            sentence = (
                f"Transaction initiated at {time_str} ({hour_of_day:.1f} hours into day), which is {direction} "
                f"relative to normal traffic hours (z-score: {z_val:+.2f}, accounting for {share_pct:.1f}% of deviation)."
            )
        else:
            display_val = f"{raw_val:.3f}"
            sentence = (
                f"Component {feat_name} is {direction} at {raw_val:.3f} versus normal baseline mean of {norm_mean:.3f} "
                f"(z-score: {z_val:+.2f}, contributing {share_pct:.1f}% of total anomaly deviation)."
            )

        reasons.append(
            {
                "feature": feat_name,
                "raw_value": display_val,
                "transformed_value": round(transformed_val, 4),
                "normal_mean": round(norm_mean, 4),
                "normal_std": round(norm_std, 4),
                "z_score": round(z_val, 2),
                "direction": direction,
                "share_pct": round(share_pct, 1),
                "sentence": sentence,
            }
        )

    return {
        "top_reasons": reasons,
        "total_deviation": round(total_deviation, 2),
        "note": PRIVACY_NOTE,
    }
