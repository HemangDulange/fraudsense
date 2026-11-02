"""Decision threshold tuning and operational alert rate metrics."""

from typing import Tuple, Dict, Any
import numpy as np

from src.evaluation.metrics import f1_score, precision_score, recall_score


def best_f1_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    num_thresholds: int = 300,
) -> Tuple[float, float]:
    """Find the anomaly score threshold that maximizes the F1-score on validation data.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels (0 or 1).
    y_scores : np.ndarray
        Predicted continuous scores.
    num_thresholds : int, default=300
        Number of candidate thresholds sampled across the score range.

    Returns
    -------
    best_threshold : float
        Optimal score cutoff.
    best_f1 : float
        Maximum achieved F1-score.
    """
    scores = np.asarray(y_scores, dtype=np.float64).ravel()
    y_t = np.asarray(y_true, dtype=np.int64).ravel()

    # Sample candidate thresholds across score percentiles
    quantiles = np.linspace(0.01, 0.999, num_thresholds)
    candidates = np.quantile(scores, quantiles)

    best_thresh = float(candidates[0])
    best_score = -1.0

    for th in candidates:
        preds = (scores >= th).astype(np.int64)
        score = f1_score(y_t, preds)
        if score > best_score:
            best_score = score
            best_thresh = float(th)

    return best_thresh, float(best_score)


def threshold_for_recall(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    target_recall: float = 0.80,
    num_thresholds: int = 500,
) -> Tuple[float, float, float]:
    """Find the threshold achieving at least target_recall with the highest precision.

    This represents the operational scenario where a fraud operations team mandates
    stopping at least 80% of all frauds while minimizing costly false alarms for normal users.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels.
    y_scores : np.ndarray
        Predicted scores.
    target_recall : float, default=0.80
        Minimum required recall for fraud detection.
    num_thresholds : int, default=500
        Number of thresholds to test.

    Returns
    -------
    chosen_threshold : float
        The selected threshold.
    achieved_recall : float
        Actual recall at this threshold.
    achieved_precision : float
        Actual precision at this threshold.
    """
    scores = np.asarray(y_scores, dtype=np.float64).ravel()
    y_t = np.asarray(y_true, dtype=np.int64).ravel()

    quantiles = np.linspace(0.001, 0.999, num_thresholds)
    candidates = np.quantile(scores, quantiles)

    # Sort descending: lower thresholds yield higher recall
    candidates = np.sort(candidates)[::-1]

    best_th = float(candidates[0])
    best_prec = -1.0
    best_rec = 0.0

    for th in candidates:
        preds = (scores >= th).astype(np.int64)
        rec = recall_score(y_t, preds)
        prec = precision_score(y_t, preds)

        if rec >= target_recall:
            if prec > best_prec:
                best_prec = prec
                best_rec = rec
                best_th = float(th)

    # Fallback if no threshold met target recall: pick lowest threshold (max recall)
    if best_prec < 0.0:
        best_th = float(candidates[-1])
        preds = (scores >= best_th).astype(np.int64)
        best_rec = recall_score(y_t, preds)
        best_prec = precision_score(y_t, preds)

    return best_th, float(best_rec), float(best_prec)


def alerts_per_10000(y_scores: np.ndarray, threshold: float) -> float:
    """Calculate expected number of fraud alerts generated per 10,000 transactions.

    Formula:
        (Number of transactions >= threshold / Total transactions) * 10,000

    This is an essential operations metric: a bank needs to know if their fraud team
    has the staff capacity to review the number of alerts flagged per 10,000 swipes.

    Parameters
    ----------
    y_scores : np.ndarray
        Predicted scores.
    threshold : float
        Decision threshold.

    Returns
    -------
    float
        Alerts per 10,000 transactions.
    """
    scores = np.asarray(y_scores, dtype=np.float64).ravel()
    if len(scores) == 0:
        return 0.0
    alert_fraction = np.mean(scores >= threshold)
    return float(alert_fraction * 10000.0)
