"""Evaluation and report figure generation script for FraudSense.

Workflow:
1. Reloads saved artifacts from artifacts/metrics/results.json and predictions.npz.
2. Recomputes test metrics from scratch and asserts exact match with results.json.
3. Generates publication-ready figures in artifacts/figures/:
   - roc_curves.png: Receiver Operating Characteristic comparison.
   - pr_curves.png: Precision-Recall curves comparison.
   - confusion_matrix.png: Detailed confusion matrix of the deployed model.
   - model_comparison.png: Side-by-side performance bar chart across models.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np

# Use headless backend for matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Add repo root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    METRICS_DIR,
    FIGURES_DIR,
)
from src.evaluation.metrics import classification_metrics
from src.evaluation.curves import roc_curve, roc_auc, precision_recall_curve, average_precision


def load_metrics_and_predictions() -> tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Load results.json and predictions.npz."""
    results_path = METRICS_DIR / "results.json"
    predictions_path = METRICS_DIR / "predictions.npz"

    if not results_path.exists():
        raise FileNotFoundError(f"Missing {results_path}. Run scripts/train_models.py first.")
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing {predictions_path}. Run scripts/train_models.py first.")

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    with np.load(predictions_path, allow_pickle=False) as npz:
        predictions = {k: npz[k] for k in npz.files}

    return results, predictions


def verify_test_metrics(results: Dict[str, Any], predictions: Dict[str, np.ndarray]) -> None:
    """Recompute test metrics from predictions and assert that they match results.json."""
    print("[INFO] Recomputing test metrics and verifying integrity...")
    y_test = predictions["y_test"]

    for model_key, model_info in results["models"].items():
        test_scores = predictions[f"{model_key}_test_scores"]
        test_preds = predictions[f"{model_key}_test_preds"]

        # Recompute
        recomputed_metrics = classification_metrics(y_test, test_preds)
        fpr, tpr, _ = roc_curve(y_test, test_scores)
        recomputed_roc_auc = roc_auc(fpr, tpr)
        recomputed_pr_auc = average_precision(y_test, test_scores)

        saved_metrics = model_info["test_metrics"]

        # Assert matching
        assert recomputed_metrics["f1"] == saved_metrics["f1"], (
            f"F1 mismatch for {model_key}: {recomputed_metrics['f1']} vs {saved_metrics['f1']}"
        )
        assert np.isclose(recomputed_roc_auc, saved_metrics["roc_auc"], atol=1e-3), (
            f"ROC-AUC mismatch for {model_key}: {recomputed_roc_auc} vs {saved_metrics['roc_auc']}"
        )
        assert np.isclose(recomputed_pr_auc, saved_metrics["pr_auc"], atol=1e-3), (
            f"PR-AUC mismatch for {model_key}: {recomputed_pr_auc} vs {saved_metrics['pr_auc']}"
        )

        print(f"  [OK] Verified metrics for {model_info['name']}")


def plot_roc_curves(results: Dict[str, Any], predictions: Dict[str, np.ndarray], output_path: Path):
    """Plot overlaid ROC curves for all models."""
    plt.figure(figsize=(9, 7))
    y_test = predictions["y_test"]

    # Baseline random guess
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.500)", alpha=0.6)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, (model_key, model_info) in enumerate(results["models"].items()):
        scores = predictions[f"{model_key}_test_scores"]
        fpr, tpr, _ = roc_curve(y_test, scores)
        auc_score = model_info["test_metrics"]["roc_auc"]
        label = f"{model_info['name']} (AUC = {auc_score:.3f})"
        plt.plot(fpr, tpr, label=label, color=colors[i % len(colors)], linewidth=2)

    plt.title("ROC Curves Comparison (Held-out Test Set)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("False Positive Rate (FPR)", fontsize=12)
    plt.ylabel("True Positive Rate (Recall / TPR)", fontsize=12)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[OK] Saved ROC curves to {output_path}")


def plot_pr_curves(results: Dict[str, Any], predictions: Dict[str, np.ndarray], output_path: Path):
    """Plot overlaid Precision-Recall curves for all models."""
    plt.figure(figsize=(9, 7))
    y_test = predictions["y_test"]
    base_fraud_rate = float(np.mean(y_test == 1))

    # Baseline no-skill line
    plt.axhline(
        y=base_fraud_rate,
        color="gray",
        linestyle="--",
        label=f"No-Skill Baseline (Rate = {base_fraud_rate:.4f})",
        alpha=0.7,
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, (model_key, model_info) in enumerate(results["models"].items()):
        scores = predictions[f"{model_key}_test_scores"]
        prec, rec, _ = precision_recall_curve(y_test, scores)
        pr_auc = model_info["test_metrics"]["pr_auc"]
        label = f"{model_info['name']} (PR-AUC = {pr_auc:.3f})"
        plt.plot(rec, prec, label=label, color=colors[i % len(colors)], linewidth=2)

    plt.title("Precision-Recall Curves (Held-out Test Set)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Recall (Fraction of Frauds Caught)", fontsize=12)
    plt.ylabel("Precision (Fraction of Alerts that are Real Fraud)", fontsize=12)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.legend(loc="upper right", frameon=True, fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[OK] Saved PR curves to {output_path}")


def plot_confusion_matrix(results: Dict[str, Any], output_path: Path):
    """Plot styled confusion matrix for the deployed model."""
    deployed_key = results["deployed_model"]
    model_info = results["models"][deployed_key]
    cm = np.array(model_info["test_metrics"]["confusion_matrix"])

    plt.figure(figsize=(7, 6))
    labels = ["Normal (0)", "Fraud (1)"]

    # Annotations with counts and percentages
    total = np.sum(cm)
    annot = np.array([
        [f"{cm[0, 0]:,d}\n({cm[0, 0]/total*100:.2f}%)", f"{cm[0, 1]:,d}\n({cm[0, 1]/total*100:.2f}%)"],
        [f"{cm[1, 0]:,d}\n({cm[1, 0]/total*100:.2f}%)", f"{cm[1, 1]:,d}\n({cm[1, 1]/total*100:.2f}%)"],
    ])

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": 13, "fontweight": "bold"},
    )

    plt.title(f"Confusion Matrix: {model_info['name']} (Deployed)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[OK] Saved Confusion Matrix to {output_path}")


def plot_model_comparison_bar(results: Dict[str, Any], output_path: Path):
    """Plot side-by-side comparison of test metrics across models."""
    models = list(results["models"].keys())
    model_names = [results["models"][m]["name"] for m in models]

    f1_scores = [results["models"][m]["test_metrics"]["f1"] for m in models]
    pr_aucs = [results["models"][m]["test_metrics"]["pr_auc"] for m in models]
    recalls = [results["models"][m]["test_metrics"]["recall"] for m in models]
    precisions = [results["models"][m]["test_metrics"]["precision"] for m in models]

    x = np.arange(len(models))
    width = 0.2

    plt.figure(figsize=(11, 6))
    plt.bar(x - 1.5 * width, precisions, width, label="Precision", color="#1f77b4")
    plt.bar(x - 0.5 * width, recalls, width, label="Recall", color="#2ca02c")
    plt.bar(x + 0.5 * width, f1_scores, width, label="F1-Score", color="#ff7f0e")
    plt.bar(x + 1.5 * width, pr_aucs, width, label="PR-AUC", color="#d62728")

    plt.title("Model Performance Comparison (Held-out Test Set)", fontsize=14, fontweight="bold", pad=12)
    plt.xticks(x, model_names, rotation=15, ha="right", fontsize=10)
    plt.ylabel("Score", fontsize=12)
    plt.ylim([0.0, 1.05])
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[OK] Saved model comparison chart to {output_path}")


def main():
    print("\n========================================================")
    print("  [FraudSense] Model Evaluation & Figure Generation")
    print("========================================================")

    results, predictions = load_metrics_and_predictions()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Assert recomputed metrics match results.json
    verify_test_metrics(results, predictions)

    # 2. Generate Figures
    print("\n[INFO] Generating publication-ready report figures...")
    plot_roc_curves(results, predictions, FIGURES_DIR / "roc_curves.png")
    plot_pr_curves(results, predictions, FIGURES_DIR / "pr_curves.png")
    plot_confusion_matrix(results, FIGURES_DIR / "confusion_matrix.png")
    plot_model_comparison_bar(results, FIGURES_DIR / "model_comparison.png")

    print("\n[OK] All figures generated successfully in artifacts/figures/!\n")


if __name__ == "__main__":
    main()
