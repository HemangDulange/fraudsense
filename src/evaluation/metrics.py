"""Classification metrics implemented completely from scratch using NumPy.

No scikit-learn or third-party ML metrics libraries are used.
"""

from typing import Dict, Any, Union
import numpy as np


def confusion_matrix(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
) -> np.ndarray:
    """Compute 2x2 confusion matrix [[TN, FP], [FN, TP]].

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth binary labels (0 = Normal, 1 = Fraud).
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels (0 = Normal, 1 = Fraud).

    Returns
    -------
    np.ndarray of shape (2, 2)
        Confusion matrix organized as:
        [[True Negatives,  False Positives],
         [False Negatives, True Positives]]
    """
    y_t = np.asarray(y_true, dtype=np.int64).ravel()
    y_p = np.asarray(y_pred, dtype=np.int64).ravel()

    if len(y_t) != len(y_p):
        raise ValueError(f"Lengths differ: len(y_true)={len(y_t)} vs len(y_pred)={len(y_p)}")

    tn = int(np.sum((y_t == 0) & (y_p == 0)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))
    tp = int(np.sum((y_t == 1) & (y_p == 1)))

    return np.array([[tn, fp], [fn, tp]], dtype=np.int64)


def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute precision: TP / (TP + FP). Returns 0.0 if denominator is 0."""
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]
    fp = cm[0, 1]
    denom = tp + fp
    return float(tp / denom) if denom > 0 else 0.0


def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute recall (sensitivity / true positive rate): TP / (TP + FN)."""
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]
    fn = cm[1, 0]
    denom = tp + fn
    return float(tp / denom) if denom > 0 else 0.0


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute specificity (true negative rate): TN / (TN + FP)."""
    cm = confusion_matrix(y_true, y_pred)
    tn = cm[0, 0]
    fp = cm[0, 1]
    denom = tn + fp
    return float(tn / denom) if denom > 0 else 0.0


def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute False Positive Rate: FP / (FP + TN) = 1 - specificity."""
    cm = confusion_matrix(y_true, y_pred)
    fp = cm[0, 1]
    tn = cm[0, 0]
    denom = fp + tn
    return float(fp / denom) if denom > 0 else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute F1 score: harmonic mean of precision and recall."""
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    denom = p + r
    return float(2.0 * p * r / denom) if denom > 0 else 0.0


def balanced_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute balanced accuracy: (recall + specificity) / 2."""
    r = recall_score(y_true, y_pred)
    s = specificity_score(y_true, y_pred)
    return float((r + s) / 2.0)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute simple accuracy: (TP + TN) / Total."""
    cm = confusion_matrix(y_true, y_pred)
    total = np.sum(cm)
    correct = cm[0, 0] + cm[1, 1]
    return float(correct / total) if total > 0 else 0.0


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute a complete dictionary of classification metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])

    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    spec = specificity_score(y_true, y_pred)
    fpr = false_positive_rate(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    return {
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "accuracy": round(acc, 5),
        "precision": round(prec, 5),
        "recall": round(rec, 5),
        "specificity": round(spec, 5),
        "false_positive_rate": round(fpr, 5),
        "f1": round(f1, 5),
        "balanced_accuracy": round(bacc, 5),
    }
