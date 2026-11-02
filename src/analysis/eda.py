"""Exploratory Data Analysis (EDA) functions for Credit Card Fraud.

All metrics, correlations, and distributions are computed from scratch
using NumPy and pandas, adhering to project constraints.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.utils.config import (
    TIME_COL,
    AMOUNT_COL,
    TARGET_COL,
    PCA_FEATURE_COLS,
)


def class_balance_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze class imbalance between normal and fraudulent transactions.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the 'Class' column.

    Returns
    -------
    Dict[str, Any]
        Counts, percentages, and imbalance ratio.
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"DataFrame must contain target column '{TARGET_COL}'.")

    y = df[TARGET_COL].to_numpy()
    total_count = len(y)
    fraud_count = int(np.sum(y == 1))
    normal_count = int(np.sum(y == 0))

    fraud_rate = float(fraud_count / total_count) if total_count > 0 else 0.0
    fraud_pct = fraud_rate * 100.0
    normal_pct = 100.0 - fraud_pct

    imbalance_ratio = (
        f"1 fraud : {int(round(normal_count / fraud_count))} normal"
        if fraud_count > 0
        else "No frauds present"
    )

    return {
        "total_transactions": total_count,
        "normal_count": normal_count,
        "fraud_count": fraud_count,
        "normal_pct": round(normal_pct, 4),
        "fraud_pct": round(fraud_pct, 4),
        "imbalance_ratio": imbalance_ratio,
    }


def amount_distribution_analysis(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compare transaction amount distribution between normal and fraud transactions.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing 'Amount' and 'Class' columns.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Summary statistics (mean, median, std, percentiles) for each class.
    """
    if AMOUNT_COL not in df.columns or TARGET_COL not in df.columns:
        raise ValueError(f"DataFrame must contain '{AMOUNT_COL}' and '{TARGET_COL}'.")

    amounts = df[AMOUNT_COL].to_numpy(dtype=np.float32)
    classes = df[TARGET_COL].to_numpy()

    def _calc_stats(arr: np.ndarray) -> Dict[str, float]:
        if len(arr) == 0:
            return {}
        return {
            "count": int(len(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }

    normal_stats = _calc_stats(amounts[classes == 0])
    fraud_stats = _calc_stats(amounts[classes == 1])

    return {
        "normal": normal_stats,
        "fraud": fraud_stats,
    }


def fraud_rate_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Compute transaction volume and fraud rate grouped by hour of the day (0-23).

    Time in the raw dataset is recorded as seconds elapsed since the first transaction.
    Hour of day is derived as: (Time % 86400) // 3600.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing 'Time' and 'Class' columns.

    Returns
    -------
    pd.DataFrame
        Table with hour, total transactions, fraud count, and fraud rate per hour.
    """
    if TIME_COL not in df.columns or TARGET_COL not in df.columns:
        raise ValueError(f"DataFrame must contain '{TIME_COL}' and '{TARGET_COL}'.")

    times = df[TIME_COL].to_numpy(dtype=np.float32)
    classes = df[TARGET_COL].to_numpy()

    # Discretize time into 24 one-hour bins (0 to 23)
    hours = ((times % 86400.0) // 3600.0).astype(int)

    rows = []
    for h in range(24):
        mask = hours == h
        total_in_hour = int(np.sum(mask))
        if total_in_hour > 0:
            fraud_in_hour = int(np.sum(classes[mask] == 1))
            normal_in_hour = total_in_hour - fraud_in_hour
            rate = float(fraud_in_hour / total_in_hour)
            rate_pct = rate * 100.0
        else:
            fraud_in_hour = 0
            normal_in_hour = 0
            rate = 0.0
            rate_pct = 0.0

        rows.append(
            {
                "hour": h,
                "total_transactions": total_in_hour,
                "normal_transactions": normal_in_hour,
                "fraud_transactions": fraud_in_hour,
                "fraud_rate": round(rate, 6),
                "fraud_rate_pct": round(rate_pct, 4),
            }
        )

    return pd.DataFrame(rows)


def correlation_with_class(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute Pearson correlation coefficient between features and target Class from scratch.

    Pearson correlation formula:
        r = sum((x - mean_x) * (y - mean_y)) / (sqrt(sum((x - mean_x)^2)) * sqrt(sum((y - mean_y)^2)))

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing features and 'Class'.
    feature_cols : Optional[List[str]], default=None
        Features to compute correlation for. Defaults to V1..V28.

    Returns
    -------
    pd.DataFrame
        DataFrame sorted by absolute correlation magnitude in descending order.
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in DataFrame.")

    cols = feature_cols if feature_cols is not None else PCA_FEATURE_COLS

    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    y_diff = y - np.mean(y)
    y_denom = np.sqrt(np.sum(y_diff**2))

    results = []
    for col in cols:
        if col not in df.columns:
            continue
        x = df[col].to_numpy(dtype=np.float64)
        x_diff = x - np.mean(x)
        x_denom = np.sqrt(np.sum(x_diff**2))

        denom = x_denom * y_denom
        if denom < 1e-12:
            r = 0.0
        else:
            r = float(np.sum(x_diff * y_diff) / denom)

        results.append(
            {
                "feature": col,
                "correlation": round(r, 6),
                "abs_correlation": round(abs(r), 6),
            }
        )

    corr_df = pd.DataFrame(results)
    if not corr_df.empty:
        corr_df = corr_df.sort_values(by="abs_correlation", ascending=False).reset_index(drop=True)

    return corr_df
