"""Unit tests for evaluation metrics, curves, and threshold tuning."""

import pytest
import numpy as np

from src.evaluation.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    specificity_score,
    false_positive_rate,
    f1_score,
    balanced_accuracy_score,
    classification_metrics,
)
from src.evaluation.curves import (
    roc_curve,
    roc_auc,
    precision_recall_curve,
    average_precision,
)
from src.evaluation.thresholds import (
    best_f1_threshold,
    threshold_for_recall,
    alerts_per_10000,
)


def test_confusion_matrix_and_basic_metrics():
    """Verify manual calculation matches metrics implementation."""
    # 5 true negatives (0, 0), 2 false positives (0, 1), 1 false negative (1, 0), 4 true positives (1, 1)
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1])

    cm = confusion_matrix(y_true, y_pred)
    assert cm[0, 0] == 5  # TN
    assert cm[0, 1] == 2  # FP
    assert cm[1, 0] == 1  # FN
    assert cm[1, 1] == 4  # TP

    prec = precision_score(y_true, y_pred)
    assert pytest.approx(prec) == 4.0 / (4.0 + 2.0)  # 4/6 = 0.6667

    rec = recall_score(y_true, y_pred)
    assert pytest.approx(rec) == 4.0 / (4.0 + 1.0)  # 4/5 = 0.80

    spec = specificity_score(y_true, y_pred)
    assert pytest.approx(spec) == 5.0 / (5.0 + 2.0)  # 5/7 = 0.7143

    fpr = false_positive_rate(y_true, y_pred)
    assert pytest.approx(fpr) == 2.0 / (5.0 + 2.0)  # 2/7 = 0.2857
    assert pytest.approx(fpr + spec) == 1.0

    f1 = f1_score(y_true, y_pred)
    expected_f1 = 2 * (prec * rec) / (prec + rec)
    assert pytest.approx(f1) == expected_f1

    bacc = balanced_accuracy_score(y_true, y_pred)
    assert pytest.approx(bacc) == (rec + spec) / 2.0

    all_metrics = classification_metrics(y_true, y_pred)
    assert all_metrics["f1"] == round(expected_f1, 5)


def test_division_by_zero_safety():
    """Verify metrics return 0.0 without errors when dividing by zero."""
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 0])

    assert precision_score(y_true, y_pred) == 0.0
    assert recall_score(y_true, y_pred) == 0.0
    assert f1_score(y_true, y_pred) == 0.0
    assert specificity_score(y_true, y_pred) == 1.0


def test_roc_and_auc():
    """Verify ROC curve calculation and trapezoidal AUC integration."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # Perfect ranking: all 1s have higher scores than all 0s
    y_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc = roc_auc(fpr, tpr)
    assert pytest.approx(auc, abs=1e-3) == 1.0

    # Inverted ranking (worst classifier)
    fpr_bad, tpr_bad, _ = roc_curve(y_true, -y_scores)
    auc_bad = roc_auc(fpr_bad, tpr_bad)
    assert pytest.approx(auc_bad, abs=1e-3) == 0.0


def test_precision_recall_and_average_precision():
    """Verify PR curve calculation and PR-AUC."""
    y_true = np.array([0, 0, 0, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])

    prec, rec, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = average_precision(y_true, y_scores)
    assert pr_auc > 0.85


def test_threshold_tuning_and_alert_rate():
    """Verify best_f1_threshold, threshold_for_recall, and alerts_per_10000."""
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    y_scores = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.8, 0.9])

    best_th, best_f1 = best_f1_threshold(y_true, y_scores)
    assert best_f1 == 1.0
    assert best_th > 0.45 and best_th <= 0.8

    # Target recall >= 0.80
    th_rec80, ach_rec, ach_prec = threshold_for_recall(y_true, y_scores, target_recall=0.80)
    assert ach_rec >= 0.80
    assert ach_prec == 1.0

    # Alerts per 10,000 transactions
    # 2 out of 10 are >= 0.8 -> 20% -> 2,000 alerts per 10,000
    alerts = alerts_per_10000(y_scores, threshold=0.8)
    assert pytest.approx(alerts) == 2000.0
