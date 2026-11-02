"""ROC and Precision-Recall curve calculations from scratch using NumPy.

Trapezoidal integration is implemented explicitly with basic NumPy vector operations
to avoid version deprecation differences between NumPy 1.x and 2.x.
"""

from typing import Tuple
import numpy as np


def roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    max_thresholds: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Receiver Operating Characteristic (ROC) curve: FPR, TPR, thresholds.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth labels (0 or 1).
    y_scores : np.ndarray
        Continuous anomaly or probability scores (higher = more suspicious).
    max_thresholds : int, default=1000
        Maximum number of discrete threshold points to evaluate (to keep
        calculations lightweight and fast).

    Returns
    -------
    fpr : np.ndarray
        False Positive Rates sorted in ascending order [0.0 to 1.0].
    tpr : np.ndarray
        True Positive Rates corresponding to each FPR.
    thresholds : np.ndarray
        Score thresholds evaluated.
    """
    y_t = np.asarray(y_true, dtype=np.int64).ravel()
    scores = np.asarray(y_scores, dtype=np.float64).ravel()

    n_pos = np.sum(y_t == 1)
    n_neg = np.sum(y_t == 0)

    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.0, 0.0])

    # Distinct sorted thresholds
    unique_scores = np.unique(scores)
    if len(unique_scores) > max_thresholds:
        # Uniform quantile downsampling of thresholds for efficiency
        quantiles = np.linspace(0.0, 1.0, max_thresholds)
        thresholds = np.quantile(unique_scores, quantiles)
    else:
        thresholds = unique_scores

    # Sort thresholds descending
    thresholds = np.sort(thresholds)[::-1]

    fpr_list = [0.0]
    tpr_list = [0.0]
    thresh_list = [thresholds[0] + 1e-6]

    for th in thresholds:
        preds = (scores >= th).astype(np.int64)
        tp = np.sum((y_t == 1) & (preds == 1))
        fp = np.sum((y_t == 0) & (preds == 1))

        tpr = tp / n_pos
        fpr = fp / n_neg

        fpr_list.append(float(fpr))
        tpr_list.append(float(tpr))
        thresh_list.append(float(th))

    # Ensure curve ends at (1.0, 1.0)
    fpr_list.append(1.0)
    tpr_list.append(1.0)
    thresh_list.append(thresholds[-1] - 1e-6)

    fpr_arr = np.array(fpr_list, dtype=np.float32)
    tpr_arr = np.array(tpr_list, dtype=np.float32)
    thresh_arr = np.array(thresh_list, dtype=np.float32)

    return fpr_arr, tpr_arr, thresh_arr


def roc_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Compute Area Under the ROC Curve (ROC-AUC) using trapezoidal integration.

    Parameters
    ----------
    fpr : np.ndarray
        False Positive Rates.
    tpr : np.ndarray
        True Positive Rates.

    Returns
    -------
    float
        Area under the ROC curve, bounded in [0.0, 1.0].
    """
    fpr_clean = np.asarray(fpr, dtype=np.float64)
    tpr_clean = np.asarray(tpr, dtype=np.float64)

    # Ensure FPR is sorted ascending
    sort_idx = np.argsort(fpr_clean)
    fpr_sorted = fpr_clean[sort_idx]
    tpr_sorted = tpr_clean[sort_idx]

    # Explicit trapezoidal rule: sum((x_i - x_{i-1}) * (y_i + y_{i-1}) / 2)
    dx = np.diff(fpr_sorted)
    avg_y = (tpr_sorted[:-1] + tpr_sorted[1:]) / 2.0
    auc = float(np.sum(dx * avg_y))

    return max(0.0, min(1.0, auc))


def precision_recall_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    max_thresholds: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Precision-Recall curve across score thresholds.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth labels (0 or 1).
    y_scores : np.ndarray
        Continuous anomaly or probability scores.
    max_thresholds : int, default=1000
        Number of thresholds to sample.

    Returns
    -------
    precision : np.ndarray
        Precision values.
    recall : np.ndarray
        Recall values.
    thresholds : np.ndarray
        Thresholds evaluated.
    """
    y_t = np.asarray(y_true, dtype=np.int64).ravel()
    scores = np.asarray(y_scores, dtype=np.float64).ravel()

    n_pos = np.sum(y_t == 1)
    if n_pos == 0:
        return np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0])

    unique_scores = np.unique(scores)
    if len(unique_scores) > max_thresholds:
        quantiles = np.linspace(0.0, 1.0, max_thresholds)
        thresholds = np.quantile(unique_scores, quantiles)
    else:
        thresholds = unique_scores

    thresholds = np.sort(thresholds)

    precision_list = []
    recall_list = []
    thresh_list = []

    for th in thresholds:
        preds = (scores >= th).astype(np.int64)
        tp = np.sum((y_t == 1) & (preds == 1))
        fp = np.sum((y_t == 0) & (preds == 1))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
        rec = float(tp / n_pos)

        precision_list.append(prec)
        recall_list.append(rec)
        thresh_list.append(float(th))

    # Add end points: recall 0 has precision 1.0
    precision_list.append(1.0)
    recall_list.append(0.0)

    return (
        np.array(precision_list, dtype=np.float32),
        np.array(recall_list, dtype=np.float32),
        np.array(thresh_list, dtype=np.float32),
    )


def average_precision(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute Precision-Recall Area Under Curve (PR-AUC) using trapezoidal integration.

    In extreme class imbalance problems (like credit card fraud), PR-AUC is far
    more informative than ROC-AUC because it focuses specifically on the rare minority class.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels.
    y_scores : np.ndarray
        Predicted anomaly or probability scores.

    Returns
    -------
    float
        PR-AUC score between 0.0 and 1.0.
    """
    precisions, recalls, _ = precision_recall_curve(y_true, y_scores)

    # Sort by recall in ascending order to integrate
    sort_idx = np.argsort(recalls)
    recalls_sorted = recalls[sort_idx]
    precisions_sorted = precisions[sort_idx]

    # Trapezoidal integration
    dr = np.diff(recalls_sorted)
    avg_p = (precisions_sorted[:-1] + precisions_sorted[1:]) / 2.0
    pr_auc = float(np.sum(dr * avg_p))

    return max(0.0, min(1.0, pr_auc))
