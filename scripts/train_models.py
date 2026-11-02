"""Model training and evaluation pipeline for FraudSense.

Workflow:
1. Load dataset (optional --sample N for lightweight quick runs).
2. Stratified train/val/test split (60% / 20% / 20%).
3. Fit Preprocessor ONLY on normal training rows (prevent data leakage).
4. For each detector, grid-search hyperparameters on VALIDATION by PR-AUC.
5. Choose optimal classification threshold on VALIDATION by maximizing F1 score.
6. Select deployed model = model with highest validation PR-AUC (test set NOT consulted).
7. Evaluate all models on TEST set exactly once.
8. Save models to artifacts/models (JSON + NPZ with allow_pickle=False).
9. Save results.json and predictions.npz to artifacts/metrics.
10. Print formatted results table and execution timing per model.
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

# Add repo root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    RAW_DATA_FILE,
    SAMPLE_DATA_FILE,
    MODELS_DIR,
    METRICS_DIR,
    RANDOM_SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
)
from src.data.loader import load_raw
from src.data.preprocessing import stratified_split, Preprocessor
from src.evaluation.metrics import classification_metrics
from src.evaluation.curves import roc_curve, roc_auc, average_precision
from src.evaluation.thresholds import (
    best_f1_threshold,
    threshold_for_recall,
    alerts_per_10000,
)
from src.utils.serialization import save_model
from src.analysis.risk import RiskScorer
from src.models.gaussian_detector import GaussianDetector
from src.models.mahalanobis_detector import MahalanobisDetector
from src.models.pca_detector import PCADetector
from src.models.isolation_forest import IsolationForestFromScratch
from src.models.logistic_regression import LogisticRegressionFromScratch


def parse_args():
    parser = argparse.ArgumentParser(description="Train FraudSense anomaly detectors and baseline.")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Subsample size N (keeps ALL frauds and N-frauds normal transactions) for fast laptop dev runs.",
    )
    return parser.parse_args()


def load_dataset_with_sampling(sample_size: int = None) -> pd.DataFrame:
    """Load raw dataset with optional stratified subsampling for laptop speed."""
    if RAW_DATA_FILE.exists():
        target_path = RAW_DATA_FILE
    elif SAMPLE_DATA_FILE.exists():
        print(f"[WARN] Raw data not found. Using sample dataset: {SAMPLE_DATA_FILE}")
        target_path = SAMPLE_DATA_FILE
    else:
        load_raw(RAW_DATA_FILE)

    print(f"[INFO] Loading data from {target_path}...")
    df = load_raw(target_path)

    if sample_size is not None and sample_size < len(df):
        fraud_mask = df["Class"] == 1
        n_frauds = int(fraud_mask.sum())
        n_normals_desired = max(10, sample_size - n_frauds)

        rng = np.random.default_rng(RANDOM_SEED)
        normal_indices = np.where(~fraud_mask)[0]
        sampled_normal_idx = rng.choice(normal_indices, size=min(n_normals_desired, len(normal_indices)), replace=False)
        fraud_idx = np.where(fraud_mask)[0]

        combined_idx = np.concatenate([sampled_normal_idx, fraud_idx])
        rng.shuffle(combined_idx)
        df = df.iloc[combined_idx].reset_index(drop=True)
        print(
            f"[INFO] Subsampled dataset to {len(df):,d} rows ({n_frauds} frauds + {len(sampled_normal_idx):,d} normal rows)."
        )

    return df


def main():
    args = parse_args()
    start_total_time = time.perf_counter()

    print("\n========================================================")
    print("  [FraudSense] Training Pipeline & Hyperparameter Tuning")
    print("========================================================")

    # 1. Load Data
    df = load_dataset_with_sampling(args.sample)
    y_all = df["Class"].to_numpy()

    # 2. Stratified Split (60% Train / 20% Val / 20% Test)
    print(f"[INFO] Splitting dataset ({TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO})...")
    train_idx, val_idx, test_idx = stratified_split(
        y_all,
        ratios=(TRAIN_RATIO, VAL_RATIO, TEST_RATIO),
        seed=RANDOM_SEED,
    )

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    y_train = df_train["Class"].to_numpy()
    y_val = df_val["Class"].to_numpy()
    y_test = df_test["Class"].to_numpy()

    print(f"  Train : {len(df_train):,d} samples ({np.sum(y_train == 1)} frauds)")
    print(f"  Val   : {len(df_val):,d} samples ({np.sum(y_val == 1)} frauds)")
    print(f"  Test  : {len(df_test):,d} samples ({np.sum(y_test == 1)} frauds)")

    # 3. Fit Preprocessor ONLY on normal training rows (No Data Leakage)
    print("[INFO] Fitting Preprocessor ONLY on normal (Class=0) training rows...")
    preprocessor = Preprocessor()
    df_train_normal = df_train[df_train["Class"] == 0]
    preprocessor.fit(df_train_normal)

    # Save preprocessor state
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    prep_state_path = MODELS_DIR / "preprocessor.json"
    with open(prep_state_path, "w", encoding="utf-8") as f:
        json.dump(preprocessor.to_state(), f, indent=2)
    print(f"[OK] Saved preprocessor state to {prep_state_path}")

    # Transform partitions
    X_train = preprocessor.transform(df_train)
    X_val = preprocessor.transform(df_val)
    X_test = preprocessor.transform(df_test)

    # Normal training subset for anomaly detectors
    X_train_normal = X_train[y_train == 0]

    # 4. Define Model Grids
    model_configs = {
        "gaussian": {
            "name": "Gaussian Detector",
            "class": GaussianDetector,
            "param_name": "var_smoothing",
            "param_grid": [1e-9, 1e-6, 1e-3],
            "supervised": False,
        },
        "mahalanobis": {
            "name": "Mahalanobis Detector",
            "class": MahalanobisDetector,
            "param_name": "reg",
            "param_grid": [1e-6, 1e-4, 1e-2],
            "supervised": False,
        },
        "pca": {
            "name": "PCA Reconstruction",
            "class": PCADetector,
            "param_name": "n_components",
            "param_grid": [5, 10, 15, 20],
            "supervised": False,
        },
        "isolation_forest": {
            "name": "Isolation Forest",
            "class": IsolationForestFromScratch,
            "param_name": "n_trees",
            "param_grid": [50, 100],
            "supervised": False,
        },
        "logistic_regression": {
            "name": "Logistic Regression (Supervised)",
            "class": LogisticRegressionFromScratch,
            "param_name": "l2_reg",
            "param_grid": [0.0, 0.01, 0.1],
            "supervised": True,
        },
    }

    results: Dict[str, Any] = {
        "dataset": {
            "total_samples": len(df),
            "train_samples": len(df_train),
            "val_samples": len(df_val),
            "test_samples": len(df_test),
            "total_frauds": int(np.sum(y_all == 1)),
            "test_frauds": int(np.sum(y_test == 1)),
        },
        "models": {},
    }

    test_predictions_npz: Dict[str, np.ndarray] = {
        "y_test": y_test.astype(np.int8),
    }

    best_val_pr_auc_overall = -1.0
    deployed_model_key = None

    # 5. Grid Search & Training Loop
    print("\n[INFO] Starting Hyperparameter Tuning on Validation Set (Criterion: PR-AUC)...")
    for key, cfg in model_configs.items():
        print(f"\n--- Training {cfg['name']} ---")
        best_param = None
        best_val_pr_auc = -1.0
        best_model = None
        best_train_time = 0.0

        for param_val in cfg["param_grid"]:
            t0 = time.perf_counter()
            if cfg["supervised"]:
                model = cfg["class"](**{cfg["param_name"]: param_val, "n_epochs": 400})
                model.fit(X_train, y_train)
            else:
                model = cfg["class"](**{cfg["param_name"]: param_val})
                model.fit(X_train_normal)
            train_duration = time.perf_counter() - t0

            # Score on validation set
            val_scores = model.anomaly_score(X_val)
            val_pr_auc = average_precision(y_val, val_scores)

            print(
                f"  [{cfg['param_name']}={param_val}] -> Val PR-AUC: {val_pr_auc:.4f} (Fit Time: {train_duration:.2f}s)"
            )

            if val_pr_auc > best_val_pr_auc:
                best_val_pr_auc = val_pr_auc
                best_param = param_val
                best_model = model
                best_train_time = train_duration

        print(f"  >> Selected {cfg['param_name']}={best_param} (Val PR-AUC: {best_val_pr_auc:.4f})")

        # 6. Find Optimal Threshold on Validation Set
        val_scores = best_model.anomaly_score(X_val)
        val_normal_scores = val_scores[y_val == 0]
        best_threshold, val_f1 = best_f1_threshold(y_val, val_scores)
        rec80_th, rec80_val_rec, rec80_val_prec = threshold_for_recall(y_val, val_scores, target_recall=0.80)

        # Fit empirical CDF RiskScorer on validation normal scores
        risk_scorer = RiskScorer(val_normal_scores, threshold=best_threshold, model_name=cfg["name"])

        print(f"  >> Optimal F1 Threshold (Val): {best_threshold:.6f} (F1: {val_f1:.4f})")

        # Check for deployed model candidate
        if best_val_pr_auc > best_val_pr_auc_overall:
            best_val_pr_auc_overall = best_val_pr_auc
            deployed_model_key = key
            deployed_risk_scorer = risk_scorer

        # 7. Evaluate on TEST Exactly Once
        t_infer_start = time.perf_counter()
        test_scores = best_model.anomaly_score(X_test)
        infer_duration = time.perf_counter() - t_infer_start

        test_preds = (test_scores >= best_threshold).astype(np.int8)
        test_metrics = classification_metrics(y_test, test_preds)

        # Compute ROC-AUC and PR-AUC on test set
        fpr, tpr, _ = roc_curve(y_test, test_scores)
        test_roc_auc = roc_auc(fpr, tpr)
        test_pr_auc = average_precision(y_test, test_scores)
        alert_rate_10k = alerts_per_10000(test_scores, best_threshold)

        test_metrics["roc_auc"] = round(test_roc_auc, 5)
        test_metrics["pr_auc"] = round(test_pr_auc, 5)
        test_metrics["alerts_per_10000"] = round(alert_rate_10k, 2)

        # 8. Save model artifact
        save_model(best_model, key, MODELS_DIR)

        # Record metrics
        results["models"][key] = {
            "name": cfg["name"],
            "best_param": {cfg["param_name"]: best_param},
            "train_time_seconds": round(best_train_time, 3),
            "inference_time_seconds": round(infer_duration, 4),
            "validation_metrics": {
                "val_pr_auc": round(best_val_pr_auc, 5),
                "val_f1": round(val_f1, 5),
                "threshold_f1": round(best_threshold, 6),
                "threshold_rec80": round(rec80_th, 6),
                "rec80_achieved_recall": round(rec80_val_rec, 4),
                "rec80_achieved_precision": round(rec80_val_prec, 4),
            },
            "test_metrics": test_metrics,
            "risk_scorer": risk_scorer.to_state(),
        }

        # Store test predictions for plotting & verification
        test_predictions_npz[f"{key}_test_scores"] = test_scores.astype(np.float32)
        test_predictions_npz[f"{key}_test_preds"] = test_preds.astype(np.int8)

    results["deployed_model"] = deployed_model_key
    print(f"\n[OK] Deployed model selected: {model_configs[deployed_model_key]['name']} (Best Validation PR-AUC)")

    # Save deployed model's risk scorer
    risk_scorer_path = MODELS_DIR / "risk_scorer.json"
    with open(risk_scorer_path, "w", encoding="utf-8") as f:
        json.dump(deployed_risk_scorer.to_state(), f, indent=2)
    print(f"[OK] Saved deployed risk scorer empirical CDF to {risk_scorer_path}")

    # 9. Save Metrics and Predictions
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = METRICS_DIR / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    predictions_path = METRICS_DIR / "predictions.npz"
    np.savez_compressed(predictions_path, **test_predictions_npz)

    print(f"[OK] Saved metrics to {results_path}")
    print(f"[OK] Saved test predictions to {predictions_path}")

    # 10. Print Formatted Results Table
    print("\n" + "=" * 96)
    print(f"{'Model':<30} | {'Val PR-AUC':<10} | {'Test F1':<8} | {'Test PR-AUC':<11} | {'Test ROC':<8} | {'Train(s)':<8}")
    print("-" * 96)
    for k, m in results["models"].items():
        deployed_flag = " *" if k == deployed_model_key else ""
        name = m["name"] + deployed_flag
        v_pr = m["validation_metrics"]["val_pr_auc"]
        t_f1 = m["test_metrics"]["f1"]
        t_pr = m["test_metrics"]["pr_auc"]
        t_roc = m["test_metrics"]["roc_auc"]
        t_time = m["train_time_seconds"]
        print(f"{name:<30} | {v_pr:<10.4f} | {t_f1:<8.4f} | {t_pr:<11.4f} | {t_roc:<8.4f} | {t_time:<8.2f}")
    print("=" * 96)
    print("(* Deployed model chosen strictly based on highest validation PR-AUC; test set was NOT consulted)")
    print(f"Total pipeline time: {time.perf_counter() - start_total_time:.2f} seconds.\n")


if __name__ == "__main__":
    main()
