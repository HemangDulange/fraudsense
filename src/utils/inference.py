"""Unified inference engine for FraudSense.

The ModelBundle class is the ONLY inference path used by both the Streamlit web app
and test suites:
    validate_columns -> preprocess -> score -> risk score and band -> explain
"""

from typing import Dict, Any, Union, List, Optional
from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.utils.config import (
    MODELS_DIR,
    METRICS_DIR,
    FEATURE_COLS,
    TIME_COL,
    AMOUNT_COL,
)
from src.data.preprocessing import Preprocessor
from src.analysis.risk import RiskScorer
from src.analysis.explain import explain
from src.utils.serialization import load_model


class ModelBundle:
    """Encapsulates all inference assets and provides the single production prediction path."""

    def __init__(
        self,
        model: Any,
        preprocessor: Preprocessor,
        risk_scorer: RiskScorer,
        model_name: str,
        results_meta: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.risk_scorer = risk_scorer
        self.model_name = model_name
        self.results_meta = results_meta or {}
        self.feature_order = list(preprocessor.feature_order)

    @classmethod
    def load(
        cls,
        models_dir: Path = MODELS_DIR,
        metrics_dir: Path = METRICS_DIR,
        model_key: Optional[str] = None,
    ) -> "ModelBundle":
        """Load preprocessor, model, and calibrated risk scorer from disk.

        Parameters
        ----------
        models_dir : Path, default=MODELS_DIR
            Directory containing model JSON + NPZ files.
        metrics_dir : Path, default=METRICS_DIR
            Directory containing results.json and predictions.npz.
        model_key : Optional[str], default=None
            Specific model to load (e.g. 'gaussian', 'mahalanobis', 'pca',
            'isolation_forest', 'logistic_regression'). If None, loads the
            deployed model recorded in results.json.

        Returns
        -------
        ModelBundle
            Ready-to-use inference bundle.
        """
        results_path = metrics_dir / "results.json"
        if not results_path.exists():
            raise FileNotFoundError(
                f"Metrics file not found at {results_path}. Please run scripts/train_models.py first."
            )

        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        target_model_key = model_key or results.get("deployed_model", "gaussian")

        # 1. Load Preprocessor
        prep_path = models_dir / "preprocessor.json"
        if not prep_path.exists():
            raise FileNotFoundError(f"Preprocessor state not found at {prep_path}")
        with open(prep_path, "r", encoding="utf-8") as f:
            preprocessor = Preprocessor.from_state(json.load(f))

        # 2. Load Model
        model = load_model(target_model_key, models_dir)

        # 3. Load Risk Scorer
        # Try loading dedicated risk_scorer.json, or extract from results.json
        risk_path = models_dir / "risk_scorer.json"
        if risk_path.exists() and (target_model_key == results.get("deployed_model")):
            with open(risk_path, "r", encoding="utf-8") as f:
                risk_scorer = RiskScorer.from_state(json.load(f))
        elif target_model_key in results.get("models", {}) and "risk_scorer" in results["models"][target_model_key]:
            risk_scorer = RiskScorer.from_state(results["models"][target_model_key]["risk_scorer"])
        else:
            # Fallback: retrieve threshold from results
            th = results["models"][target_model_key]["validation_metrics"]["threshold_f1"]
            risk_scorer = RiskScorer(np.linspace(0, 100, 100), threshold=th, model_name=target_model_key)

        model_display_name = results.get("models", {}).get(target_model_key, {}).get("name", target_model_key)

        return cls(
            model=model,
            preprocessor=preprocessor,
            risk_scorer=risk_scorer,
            model_name=model_display_name,
            results_meta=results,
        )

    def validate_columns(self, data: Union[pd.DataFrame, Dict[str, Any], pd.Series]) -> None:
        """Validate that all 30 expected feature columns are present."""
        if isinstance(data, pd.DataFrame):
            cols = set(data.columns)
        elif isinstance(data, (dict, pd.Series)):
            cols = set(data.keys())
        else:
            return  # Raw numpy arrays rely on length check in preprocessor

        missing = [c for c in self.feature_order if c not in cols]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")

    def predict_one(
        self,
        row: Union[pd.Series, pd.DataFrame, Dict[str, float], np.ndarray],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Execute the complete single-transaction inference pipeline:
        validate_columns -> preprocess -> score -> risk score and band -> explain
        """
        # 1. Validate Columns
        self.validate_columns(row)

        # 2. Format row as 1-row DataFrame
        if isinstance(row, dict):
            df_row = pd.DataFrame([row])[self.feature_order]
        elif isinstance(row, pd.Series):
            df_row = pd.DataFrame([row.to_dict()])[self.feature_order]
        elif isinstance(row, pd.DataFrame):
            df_row = row.head(1)[self.feature_order]
        elif isinstance(row, np.ndarray):
            df_row = pd.DataFrame(row.reshape(1, -1), columns=self.feature_order)
        else:
            raise TypeError(f"Unsupported input row type: {type(row)}")

        # 3. Preprocess
        X_prep = self.preprocessor.transform(df_row)

        # 4. Score
        raw_score = float(self.model.anomaly_score(X_prep)[0])

        # 5. Risk score and band
        risk_result = self.risk_scorer.assess_risk(raw_score)

        # 6. Explain
        explain_result = explain(df_row, preprocessor=self.preprocessor, top_k=top_k)

        return {
            "model_name": self.model_name,
            "raw_score": risk_result["raw_score"],
            "risk_score": risk_result["risk_score"],
            "risk_band": risk_result["risk_band"],
            "action": risk_result["action"],
            "badge_color": risk_result["badge_color"],
            "deployed_threshold": risk_result["deployed_threshold"],
            "disclaimer": risk_result["disclaimer"],
            "top_reasons": explain_result["top_reasons"],
            "total_deviation": explain_result["total_deviation"],
            "privacy_note": explain_result["note"],
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score a batch of transactions and return table with scores, risk bands, and actions."""
        self.validate_columns(df)

        df_input = df[self.feature_order].copy()
        X_prep = self.preprocessor.transform(df_input)

        # Score
        raw_scores = self.model.anomaly_score(X_prep)

        # Percentile rank
        risk_scores = self.risk_scorer.percentile_rank(raw_scores)

        # Vectorized Risk Bands & Actions
        th = self.risk_scorer.threshold
        is_high = raw_scores >= th
        is_med = (risk_scores >= 90.0) & (~is_high)

        bands = np.where(is_high, "High", np.where(is_med, "Medium", "Low"))
        actions = np.where(
            is_high,
            "Hold and send to manual review",
            np.where(is_med, "Ask for OTP verification", "Approve"),
        )

        out_df = df.copy()
        out_df["anomaly_score"] = np.round(raw_scores, 4)
        out_df["risk_score"] = np.round(risk_scores, 2)
        out_df["risk_band"] = bands
        out_df["suggested_action"] = actions

        return out_df
