"""Unit tests for risk assessment and feature explanation modules."""

import pytest
import numpy as np
import pandas as pd

from src.analysis.risk import RiskScorer
from src.analysis.explain import explain, PRIVACY_NOTE
from src.data.preprocessing import Preprocessor
from src.utils.config import ALL_COLS, FEATURE_COLS


@pytest.fixture
def dummy_preprocessor():
    """Create a fitted preprocessor on synthetic normal data."""
    rng = np.random.default_rng(42)
    n = 200
    data = {f"V{i}": rng.normal(0, 1, n).astype(np.float32) for i in range(1, 29)}
    data["Time"] = rng.uniform(28800, 72000, n).astype(np.float32)  # Normal 8am - 8pm
    data["Amount"] = rng.exponential(45.0, n).astype(np.float32)
    data["Class"] = np.zeros(n, dtype=np.int8)

    df = pd.DataFrame(data)[ALL_COLS]
    prep = Preprocessor()
    prep.fit(df)
    return prep


def test_risk_scorer_bands_and_disclaimer():
    """Verify empirical CDF ranking, band boundaries, and decision support disclaimer."""
    # Synthetic normal scores from 10.0 to 100.0
    val_normal_scores = np.linspace(10.0, 100.0, 1000, dtype=np.float32)
    threshold = 95.0  # 95th percentile cutoff

    scorer = RiskScorer(val_normal_scores, threshold=threshold)

    # 1. Low risk: score of 50.0 (near median, ~50th percentile)
    res_low = scorer.assess_risk(50.0)
    assert res_low["risk_band"] == "Low"
    assert res_low["action"] == "Approve"
    assert res_low["risk_score"] < 90.0
    assert res_low["disclaimer"] == "Decision support, not a verdict"

    # 2. Medium risk: score of 92.0 (around 92nd percentile, but below threshold 95.0)
    res_med = scorer.assess_risk(92.0)
    assert res_med["risk_band"] == "Medium"
    assert res_med["action"] == "Ask for OTP verification"
    assert res_med["risk_score"] >= 90.0

    # 3. High risk: score >= threshold (e.g. 96.0)
    res_high = scorer.assess_risk(96.0)
    assert res_high["risk_band"] == "High"
    assert res_high["action"] == "Hold and send to manual review"

    # 4. State roundtrip serialization
    state = scorer.to_state()
    reconstructed = RiskScorer.from_state(state)
    assert reconstructed.assess_risk(96.0)["risk_band"] == "High"


def test_explain_top_5_reasons(dummy_preprocessor):
    """Verify explain returns top 5 deviations, direction, share, and sentences."""
    # Create an anomalous transaction row:
    # Extreme Amount, unusual Time, and anomalous V14 and V10
    row = {f"V{i}": 0.0 for i in range(1, 29)}
    row["V14"] = -8.5   # Massive negative z-score
    row["V10"] = -6.0   # Large negative z-score
    row["Amount"] = 4500.0  # Massive spending spike
    row["Time"] = 10800.0   # 3:00 AM (unusual time)

    result = explain(row, preprocessor=dummy_preprocessor, top_k=5)

    assert "top_reasons" in result
    assert "total_deviation" in result
    assert "note" in result
    assert result["note"] == PRIVACY_NOTE

    reasons = result["top_reasons"]
    assert len(reasons) == 5

    # Exactly 5 reasons sorted by |z| descending
    z_abs = [abs(r["z_score"]) for r in reasons]
    assert z_abs == sorted(z_abs, reverse=True)

    # Top reasons should be the extreme ones: Amount, V14, V10
    top_feature_names = [r["feature"] for r in reasons]
    assert "V14" in top_feature_names
    assert "Amount" in top_feature_names

    # Check structure of the top reason
    top = reasons[0]
    assert "feature" in top
    assert "raw_value" in top
    assert "normal_mean" in top
    assert "normal_std" in top
    assert "z_score" in top
    assert top["direction"] in ("unusually high", "unusually low")
    assert top["share_pct"] > 0.0
    assert len(top["sentence"]) > 10
