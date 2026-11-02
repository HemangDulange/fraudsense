"""End-to-end integration tests using ModelBundle as the single inference path."""

import pytest
import numpy as np
import pandas as pd

from src.utils.config import MODELS_DIR, METRICS_DIR, SAMPLE_DATA_FILE
from src.utils.inference import ModelBundle
from src.data.loader import load_raw


def test_model_bundle_single_and_batch_inference():
    """Verify ModelBundle loads saved artifacts and executes the full inference pipeline."""
    results_path = METRICS_DIR / "results.json"
    if not results_path.exists():
        pytest.skip("Training artifacts not found; skipping integration test.")

    # 1. Load ModelBundle
    bundle = ModelBundle.load()
    assert bundle.model is not None
    assert bundle.preprocessor is not None
    assert bundle.risk_scorer is not None

    # 2. Test single transaction prediction
    sample_row = {col: 0.0 for col in bundle.feature_order}
    sample_row["Amount"] = 250.0
    sample_row["Time"] = 43200.0  # 12:00 PM

    single_res = bundle.predict_one(sample_row, top_k=5)
    assert "risk_score" in single_res
    assert "risk_band" in single_res
    assert "action" in single_res
    assert single_res["disclaimer"] == "Decision support, not a verdict"
    assert len(single_res["top_reasons"]) == 5
    assert single_res["risk_band"] in ("Low", "Medium", "High")

    # 3. Test missing column validation raises ValueError
    incomplete_row = {col: 0.0 for col in bundle.feature_order[1:]}  # Missing 'Time'
    with pytest.raises(ValueError) as excinfo:
        bundle.predict_one(incomplete_row)
    assert "Time" in str(excinfo.value)

    # 4. Test batch scoring
    if SAMPLE_DATA_FILE.exists():
        demo_df = load_raw(SAMPLE_DATA_FILE)
        scored_df = bundle.predict_batch(demo_df)

        assert "anomaly_score" in scored_df.columns
        assert "risk_score" in scored_df.columns
        assert "risk_band" in scored_df.columns
        assert "suggested_action" in scored_df.columns
        assert len(scored_df) == len(demo_df)
