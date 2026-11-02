"""Unit tests for Streamlit application components and helper functions."""

import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app import (
    get_model_bundle,
    get_metrics_and_predictions,
    create_risk_gauge,
    create_reasons_chart,
)
from src.utils.inference import ModelBundle


def test_app_model_bundle_loader():
    """Verify get_model_bundle successfully loads the deployed ModelBundle."""
    bundle = get_model_bundle()
    assert bundle is not None
    assert isinstance(bundle, ModelBundle)
    assert bundle.model is not None
    assert bundle.preprocessor is not None
    assert bundle.risk_scorer is not None


def test_app_metrics_loader():
    """Verify get_metrics_and_predictions loads results dictionary and predictions arrays."""
    results, predictions = get_metrics_and_predictions()
    assert results is not None
    assert predictions is not None
    assert "models" in results
    assert "deployed_model" in results
    assert "y_test" in predictions


def test_create_risk_gauge():
    """Verify create_risk_gauge generates a valid Plotly Figure."""
    fig = create_risk_gauge(risk_score=78.5, threshold_pct=95.0)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "indicator"
    assert fig.data[0].value == 78.5


def test_create_reasons_chart():
    """Verify create_reasons_chart generates a valid Plotly horizontal bar chart."""
    reasons = [
        {"feature": "Amount", "share_pct": 35.2, "direction": "unusually high"},
        {"feature": "V14", "share_pct": 28.1, "direction": "unusually low"},
        {"feature": "V10", "share_pct": 18.5, "direction": "unusually low"},
        {"feature": "Time", "share_pct": 10.2, "direction": "unusually low"},
        {"feature": "V12", "share_pct": 8.0, "direction": "unusually low"},
    ]
    fig = create_reasons_chart(reasons)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "bar"
    assert len(fig.data[0].x) == 5


def test_app_smoke_test_with_apptest():
    """Smoke test running app.py using streamlit.testing.v1.AppTest."""
    from streamlit.testing.v1 import AppTest
    from src.utils.config import PROJECT_ROOT

    app_path = PROJECT_ROOT / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30).run()
    assert not at.exception
