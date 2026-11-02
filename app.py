"""FraudSense: Credit Card Fraud Detection Streamlit Dashboard.

An interactive risk-scoring dashboard for real-time anomaly detection,
feature explanations, and fraud operations decision support.
Built entirely from scratch with NumPy.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    MODELS_DIR,
    METRICS_DIR,
    FIGURES_DIR,
    SAMPLE_DATA_FILE,
    FEATURE_COLS,
    TIME_COL,
    AMOUNT_COL,
)
from src.utils.inference import ModelBundle
from src.data.loader import load_raw
from src.analysis.eda import (
    class_balance_analysis,
    amount_distribution_analysis,
    fraud_rate_by_hour,
    correlation_with_class,
)
from src.evaluation.metrics import classification_metrics
from src.evaluation.thresholds import alerts_per_10000


# =====================================================================
# Page Configuration
# =====================================================================
st.set_page_config(
    page_title="FraudSense — Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================================
# Cached Resource Loader
# =====================================================================
@st.cache_resource
def get_model_bundle() -> Optional[ModelBundle]:
    """Load model bundle from saved artifacts using st.cache_resource."""
    try:
        return ModelBundle.load()
    except Exception as e:
        return None


@st.cache_data
def get_demo_transactions() -> Optional[pd.DataFrame]:
    """Load sample transactions for demonstration."""
    if SAMPLE_DATA_FILE.exists():
        return load_raw(SAMPLE_DATA_FILE)
    return None


@st.cache_data
def get_metrics_and_predictions() -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, np.ndarray]]]:
    """Load results.json and predictions.npz."""
    res_path = METRICS_DIR / "results.json"
    pred_path = METRICS_DIR / "predictions.npz"
    if not (res_path.exists() and pred_path.exists()):
        return None, None

    with open(res_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    with np.load(pred_path, allow_pickle=False) as npz:
        predictions = {k: npz[k] for k in npz.files}
    return results, predictions


# =====================================================================
# Error Screen for Missing Artifacts
# =====================================================================
def render_missing_artifacts_warning():
    st.error("⚠️ Model Artifacts Not Found!")
    st.markdown(
        """
        The FraudSense application is strictly configured to **never train models at startup**.
        Trained model weights and evaluation metrics must be generated beforehand.
        
        ### Required Setup Steps:
        Please open your terminal in the `fraudsense/` project root directory and execute:
        
        ```bash
        # 1. Train anomaly detectors and baseline model
        python scripts/train_models.py
        
        # (Optional: Use --sample N for lightweight quick runs on low-spec laptops)
        # python scripts/train_models.py --sample 5000
        
        # 2. Evaluate models and generate report assets
        python scripts/evaluate_models.py
        ```
        
        Once completed, refresh this page to access the interactive dashboard.
        """
    )
    st.stop()


# =====================================================================
# Plotly Risk Gauge Helper
# =====================================================================
def create_risk_gauge(risk_score: float, threshold_pct: float) -> go.Figure:
    """Create a customized risk score gauge using Plotly."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_score,
            number={"suffix": "%", "font": {"size": 42, "color": "#1B2A3A"}},
            title={"text": "<b>Fraud Risk Percentile</b>", "font": {"size": 18, "color": "#1B2A3A"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#1B2A3A"},
                "bar": {"color": "#E4572E" if risk_score >= threshold_pct else ("#F5A623" if risk_score >= 90 else "#2ECC71")},
                "bgcolor": "#FFFFFF",
                "borderwidth": 1,
                "bordercolor": "#E0E0E0",
                "steps": [
                    {"range": [0, 90], "color": "#E8F8F5"},
                    {"range": [90, threshold_pct], "color": "#FEF9E7"},
                    {"range": [threshold_pct, 100], "color": "#FDEDEC"},
                ],
                "threshold": {
                    "line": {"color": "#D9381E", "width": 4},
                    "thickness": 0.85,
                    "value": threshold_pct,
                },
            },
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "sans-serif"},
    )
    return fig


# =====================================================================
# Top 5 Reasons Bar Chart Helper
# =====================================================================
def create_reasons_chart(reasons: list) -> go.Figure:
    """Create horizontal bar chart showing top 5 feature deviation contributions."""
    features = [r["feature"] for r in reversed(reasons)]
    shares = [r["share_pct"] for r in reversed(reasons)]
    colors = ["#E4572E" if r["direction"] == "unusually high" else "#3498DB" for r in reversed(reasons)]

    fig = go.Figure(
        go.Bar(
            x=shares,
            y=features,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{s:.1f}%" for s in shares],
            textposition="auto",
        )
    )
    fig.update_layout(
        title="<b>Top Feature Deviations (% Share of Total Abnormality)</b>",
        xaxis_title="Share of Anomaly Deviation (%)",
        yaxis_title="Feature",
        height=260,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# =====================================================================
# Main Application Logic
# =====================================================================
def main():
    bundle = get_model_bundle()
    results, predictions = get_metrics_and_predictions()

    # Sidebar Navigation
    st.sidebar.title("🛡️ FraudSense")
    st.sidebar.caption("Credit Card Anomaly Detection")

    if bundle is None or results is None:
        render_missing_artifacts_warning()

    st.sidebar.markdown(f"**Deployed Engine:**\n`{bundle.model_name}`")
    st.sidebar.markdown(f"**Decision Threshold:**\n`{bundle.risk_scorer.threshold:.4f}`")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        [
            "📊 Dashboard",
            "🔍 Check a Transaction",
            "📁 Batch Scoring",
            "🔬 Model Lab",
            "🎚️ Threshold Simulator",
            "📈 Data Explorer",
            "ℹ️ About & Methodology",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Pure NumPy ML Engine**\nZero runtime training · Sub-millisecond inference.")

    # -------------------------------------------------------------
    # Page 1: Dashboard
    # -------------------------------------------------------------
    if page == "📊 Dashboard":
        st.title("📊 FraudSense Operations Dashboard")
        st.markdown(
            "Executive overview of credit card fraud detection performance and deployed model KPIs."
        )

        dataset_meta = results.get("dataset", {})
        deployed_key = results.get("deployed_model")
        deployed_info = results.get("models", {}).get(deployed_key, {})
        test_m = deployed_info.get("test_metrics", {})

        # Key Metrics Row
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Evaluated Test Set", f"{dataset_meta.get('test_samples', 0):,d}")
        col2.metric("Test Frauds", f"{dataset_meta.get('test_frauds', 0):,d}")
        col3.metric("Test PR-AUC", f"{test_m.get('pr_auc', 0.0):.4f}")
        col4.metric("Test Recall", f"{test_m.get('recall', 0.0) * 100:.1f}%")
        col5.metric("Precision", f"{test_m.get('precision', 0.0) * 100:.1f}%")

        st.markdown("---")

        # Operational Summary Card
        st.subheader("🛡️ Operational Posture & Alert Rate")
        alert_col1, alert_col2 = st.columns([1, 2])
        with alert_col1:
            st.metric(
                label="Alerts per 10,000 Swipes",
                value=f"{test_m.get('alerts_per_10000', 0.0):.1f}",
                help="Projected review volume per 10,000 transactions at the current operating threshold.",
            )
            st.metric(label="False Positive Rate", value=f"{test_m.get('false_positive_rate', 0.0) * 100:.3f}%")
            st.metric(label="Test F1-Score", value=f"{test_m.get('f1', 0.0):.4f}")

        with alert_col2:
            st.markdown(
                f"""
                ### Active Production Model: **{deployed_info.get('name', 'N/A')}**
                - **Hyperparameters:** `{deployed_info.get('best_param', {})}`
                - **Threshold Selection:** Optimized on validation set to maximize F1 (`{deployed_info.get('validation_metrics', {}).get('threshold_f1', 0.0):.4f}`).
                - **Selection Policy:** Chosen strictly on validation set PR-AUC (**held-out test set was NOT consulted**).
                - **Inference Latency:** `{deployed_info.get('inference_time_seconds', 0.0):.4f}s` per batch evaluation.
                """
            )

        st.markdown("---")
        st.subheader("📈 Key Evaluation Figures")
        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            roc_path = FIGURES_DIR / "roc_curves.png"
            if roc_path.exists():
                st.image(str(roc_path), caption="Receiver Operating Characteristic (ROC)", use_container_width=True)
        with fig_col2:
            pr_path = FIGURES_DIR / "pr_curves.png"
            if pr_path.exists():
                st.image(str(pr_path), caption="Precision-Recall Curves (PR-AUC Focus)", use_container_width=True)

    # -------------------------------------------------------------
    # Page 2: Check a Transaction
    # -------------------------------------------------------------
    elif page == "🔍 Check a Transaction":
        st.title("🔍 Real-Time Transaction Risk Assessment")
        st.markdown(
            "Test live transactions, inspect empirical risk percentile scores, and explore plain-English explanations."
        )

        demo_df = get_demo_transactions()

        # Session state for inputs
        if "input_row" not in st.session_state:
            st.session_state.input_row = {col: 0.0 for col in bundle.feature_order}
            st.session_state.input_row["Amount"] = 45.0
            st.session_state.input_row["Time"] = 36000.0

        # Quick Load Buttons
        btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
        with btn_col1:
            if st.button("🟢 Load Random Normal Sample", use_container_width=True):
                if demo_df is not None:
                    normals = demo_df[demo_df["Class"] == 0]
                    if not normals.empty:
                        sampled = normals.sample(1, random_state=int(np.random.randint(10000))).iloc[0]
                        st.session_state.input_row = {c: float(sampled[c]) for c in bundle.feature_order}
                        st.rerun()

        with btn_col2:
            if st.button("🔴 Load Random Fraud Sample", use_container_width=True):
                if demo_df is not None:
                    frauds = demo_df[demo_df["Class"] == 1]
                    if not frauds.empty:
                        sampled = frauds.sample(1, random_state=int(np.random.randint(10000))).iloc[0]
                        st.session_state.input_row = {c: float(sampled[c]) for c in bundle.feature_order}
                        st.rerun()

        with btn_col3:
            st.caption("ℹ️ *demo sample: frauds over-represented for demonstration, not an evaluation set*")

        st.markdown("---")

        # Input Form
        with st.form("transaction_form"):
            st.subheader("Transaction Parameters")
            col_amt, col_time = st.columns(2)

            with col_amt:
                current_amt = float(st.session_state.input_row.get("Amount", 50.0))
                input_amount = st.number_input(
                    "Transaction Amount ($ USD)",
                    min_value=0.0,
                    max_value=50000.0,
                    value=max(0.0, current_amt),
                    step=5.0,
                    format="%.2f",
                )

            with col_time:
                current_time = float(st.session_state.input_row.get("Time", 36000.0))
                input_time = st.number_input(
                    "Elapsed Time (Seconds)",
                    min_value=0.0,
                    max_value=172800.0,
                    value=max(0.0, current_time),
                    step=60.0,
                    format="%.1f",
                    help="Seconds elapsed since the start of transaction logging (approx 2 days total).",
                )
                hour_val = (input_time % 86400.0) / 3600.0
                st.caption(f"🕒 Converted Hour of Day: **{hour_val:.2f} hours** ({int(hour_val):02d}:{int((hour_val%1)*60):02d})")

            # Expander for V1..V28
            with st.expander("🛠️ Inspect & Edit Anonymized Features (V1 through V28)", expanded=False):
                st.caption("PCA components obtained from European cardholder transactions (anonymized for privacy).")
                v_cols = st.columns(4)
                v_inputs = {}
                for idx, v_feat in enumerate([f"V{i}" for i in range(1, 29)]):
                    target_col = v_cols[idx % 4]
                    val = float(st.session_state.input_row.get(v_feat, 0.0))
                    v_inputs[v_feat] = target_col.number_input(
                        v_feat,
                        value=val,
                        step=0.1,
                        format="%.4f",
                        key=f"input_{v_feat}",
                    )

            submitted = st.form_submit_button("⚡ Evaluate Transaction Risk", type="primary", use_container_width=True)

        # Update session row on submission
        transaction_dict = {col: v_inputs.get(col, 0.0) for col in bundle.feature_order}
        transaction_dict["Amount"] = input_amount
        transaction_dict["Time"] = input_time

        # Predict
        result = bundle.predict_one(transaction_dict, top_k=5)

        st.markdown("---")
        st.subheader("Risk Scorecard & Decision Support")

        # Assessment Display
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            # Risk Gauge
            gauge_fig = create_risk_gauge(result["risk_score"], result["deployed_threshold_percentile"])
            st.plotly_chart(gauge_fig, use_container_width=True)

        with res_col2:
            st.markdown(f"#### Status: **{result['risk_band'].upper()} RISK**")

            badge_style = {
                "Low": "background-color: #E8F8F5; color: #117A65; border: 1px solid #A3E4D7; border-radius: 8px; padding: 12px;",
                "Medium": "background-color: #FEF9E7; color: #B7950B; border: 1px solid #F9E79F; border-radius: 8px; padding: 12px;",
                "High": "background-color: #FDEDEC; color: #922B21; border: 1px solid #F5B7B1; border-radius: 8px; padding: 12px;",
            }[result["risk_band"]]

            st.markdown(
                f"""
                <div style="{badge_style}">
                    <h3 style="margin: 0; padding-bottom: 6px;">Recommended Action: {result['action']}</h3>
                    <p style="margin: 0; font-size: 14px;"><b>Raw Anomaly Score:</b> {result['raw_score']:.4f} (Threshold: {result['deployed_threshold']:.4f})</p>
                    <p style="margin: 0; font-size: 14px;"><b>Percentile Rank:</b> {result['risk_score']:.1f}% among normal transactions</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.caption(f"⚠️ *{result['disclaimer']}*")

        st.markdown("---")
        st.subheader("🔍 Top 5 Feature Deviation Explanations")

        exp_col1, exp_col2 = st.columns([1, 1])
        with exp_col1:
            reasons_chart = create_reasons_chart(result["top_reasons"])
            st.plotly_chart(reasons_chart, use_container_width=True)

        with exp_col2:
            st.markdown("**Plain-English Factor Breakdown:**")
            for i, reason in enumerate(result["top_reasons"], start=1):
                icon = "🔺" if reason["direction"] == "unusually high" else "🔻"
                st.markdown(f"**{i}. {reason['feature']}** {icon} ({reason['share_pct']:.1f}% contribution)")
                st.caption(reason["sentence"])

        st.info(result["privacy_note"])

    # -------------------------------------------------------------
    # Page 3: Batch Scoring
    # -------------------------------------------------------------
    elif page == "📁 Batch Scoring":
        st.title("📁 Batch Transaction Scoring")
        st.markdown(
            "Upload a batch CSV file to score multiple transactions simultaneously. "
            "Results include raw anomaly scores, empirical risk percentiles, risk bands, and recommended actions."
        )

        demo_df = get_demo_transactions()
        if demo_df is not None:
            csv_bytes = demo_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Demo Sample CSV",
                data=csv_bytes,
                file_name="demo_transactions_sample.csv",
                mime="text/csv",
                help="demo sample: frauds over-represented for demonstration, not an evaluation set",
            )
            st.caption("ℹ️ *demo sample: frauds over-represented for demonstration, not an evaluation set*")

        uploaded_file = st.file_uploader("Upload CSV file containing transactions", type=["csv"])

        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(batch_df):,d} transactions from uploaded CSV.")

                with st.spinner("Scoring transactions with ModelBundle..."):
                    scored_df = bundle.predict_batch(batch_df)

                # Summary Statistics
                band_counts = scored_df["risk_band"].value_counts()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Scored", f"{len(scored_df):,d}")
                c2.metric("Low Risk (Approve)", f"{band_counts.get('Low', 0):,d}")
                c3.metric("Medium Risk (OTP)", f"{band_counts.get('Medium', 0):,d}")
                c4.metric("High Risk (Review)", f"{band_counts.get('High', 0):,d}")

                # Display table
                st.dataframe(
                    scored_df[["Time", "Amount", "anomaly_score", "risk_score", "risk_band", "suggested_action"]].head(100),
                    use_container_width=True,
                )
                if len(scored_df) > 100:
                    st.caption(f"Showing first 100 of {len(scored_df):,d} rows.")

                # Download Scored CSV
                scored_csv = scored_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="💾 Download Fully Scored Results CSV",
                    data=scored_csv,
                    file_name="scored_fraud_transactions.csv",
                    mime="text/csv",
                    type="primary",
                )

            except Exception as e:
                st.error(f"Error during batch scoring: {e}")

    # -------------------------------------------------------------
    # Page 4: Model Lab
    # -------------------------------------------------------------
    elif page == "🔬 Model Lab":
        st.title("🔬 Model Lab & Performance Comparison")
        st.markdown(
            "Side-by-side comparison of all 5 algorithms implemented from scratch with pure NumPy. "
            "Evaluated strictly on held-out test data."
        )

        models_data = results.get("models", {})
        table_rows = []
        for k, m in models_data.items():
            param_str = ", ".join([f"{pk}={pv}" for pk, pv in m.get("best_param", {}).items()])
            tm = m.get("test_metrics", {})
            vm = m.get("validation_metrics", {})
            is_deployed = k == results.get("deployed_model")
            table_rows.append(
                {
                    "Model": m.get("name") + (" 🏆 (Deployed)" if is_deployed else ""),
                    "Hyperparameters": param_str,
                    "Val PR-AUC": vm.get("val_pr_auc", 0.0),
                    "Test PR-AUC": tm.get("pr_auc", 0.0),
                    "Test F1": tm.get("f1", 0.0),
                    "Test Recall": tm.get("recall", 0.0),
                    "Test Precision": tm.get("precision", 0.0),
                    "Test ROC-AUC": tm.get("roc_auc", 0.0),
                    "Train Time (s)": m.get("train_time_seconds", 0.0),
                }
            )

        comp_df = pd.DataFrame(table_rows)
        st.dataframe(comp_df, use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Comparative Evaluation Charts")
        fig_path = FIGURES_DIR / "model_comparison.png"
        if fig_path.exists():
            st.image(str(fig_path), caption="Model Performance Breakdown on Held-out Test Set", use_container_width=True)

        col_roc, col_cm = st.columns(2)
        with col_roc:
            roc_path = FIGURES_DIR / "roc_curves.png"
            if roc_path.exists():
                st.image(str(roc_path), caption="ROC Curves", use_container_width=True)
        with col_cm:
            cm_path = FIGURES_DIR / "confusion_matrix.png"
            if cm_path.exists():
                st.image(str(cm_path), caption="Deployed Model Confusion Matrix", use_container_width=True)

    # -------------------------------------------------------------
    # Page 5: Threshold Simulator
    # -------------------------------------------------------------
    elif page == "🎚️ Threshold Simulator":
        st.title("🎚️ Decision Threshold Simulator")
        st.markdown(
            "Simulate the operational trade-offs of adjusting the fraud detection threshold live on test data. "
            "Observe the delicate balance between catching frauds vs. creating false alarms for human reviewers."
        )

        deployed_key = results.get("deployed_model")
        deployed_info = results.get("models", {}).get(deployed_key, {})
        default_th = float(deployed_info.get("validation_metrics", {}).get("threshold_f1", 0.5))

        test_scores = predictions.get(f"{deployed_key}_test_scores")
        y_test = predictions.get("y_test")

        if test_scores is not None and y_test is not None:
            min_score = float(np.min(test_scores))
            max_score = float(np.max(test_scores))

            sim_threshold = st.slider(
                "Anomaly Score Decision Threshold",
                min_value=min_score,
                max_value=max_score,
                value=float(np.clip(default_th, min_score, max_score)),
                step=(max_score - min_score) / 200.0,
                format="%.4f",
            )

            preds = (test_scores >= sim_threshold).astype(int)
            metrics = classification_metrics(y_test, preds)
            alert_rate = alerts_per_10000(test_scores, sim_threshold)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Precision", f"{metrics['precision'] * 100:.1f}%")
            m2.metric("Recall (Frauds Caught)", f"{metrics['recall'] * 100:.1f}%")
            m3.metric("Missed Frauds (FN)", f"{metrics['false_negatives']:,d}")
            m4.metric("Alerts per 10k Swipes", f"{alert_rate:.1f}")

            st.markdown("---")
            st.markdown(
                f"""
                ### Operational Impact Analysis:
                - **Fraud Catch Rate:** At threshold `{sim_threshold:.4f}`, the system catches **{metrics['true_positives']} of {metrics['true_positives'] + metrics['false_negatives']}** frauds.
                - **Customer Friction:** **{metrics['false_positives']}** legitimate transactions are falsely flagged.
                - **Review Capacity:** Operations receives approximately **{alert_rate:.1f} fraud alerts for every 10,000 transactions processed**.
                """
            )

    # -------------------------------------------------------------
    # Page 6: Data Explorer
    # -------------------------------------------------------------
    elif page == "📈 Data Explorer":
        st.title("📈 Exploratory Data Analysis & Patterns")
        st.markdown("Understand the statistical distributions, temporal patterns, and feature correlations with fraud.")

        demo_df = get_demo_transactions()
        if demo_df is not None:
            tab1, tab2, tab3 = st.tabs(["💰 Amount Distribution", "🕒 Temporal Patterns", "🔗 Feature Correlations"])

            with tab1:
                st.subheader("Transaction Amount: Normal vs. Fraudulent")
                amt_analysis = amount_distribution_analysis(demo_df)
                c_norm, c_fr = st.columns(2)
                with c_norm:
                    st.metric("Normal Median Amount", f"${amt_analysis['normal'].get('median', 0.0):.2f}")
                    st.metric("Normal 95th Percentile", f"${amt_analysis['normal'].get('p95', 0.0):.2f}")
                with c_fr:
                    st.metric("Fraud Median Amount", f"${amt_analysis['fraud'].get('median', 0.0):.2f}")
                    st.metric("Fraud 95th Percentile", f"${amt_analysis['fraud'].get('p95', 0.0):.2f}")

                fig_amt = px.histogram(
                    demo_df,
                    x="Amount",
                    color="Class",
                    nbins=40,
                    log_y=True,
                    title="Transaction Amount Distribution (Log Scale)",
                    color_discrete_map={0: "#2ECC71", 1: "#E4572E"},
                )
                st.plotly_chart(fig_amt, use_container_width=True)

            with tab2:
                st.subheader("Hourly Fraud Rate Across 24-Hour Day")
                hourly_df = fraud_rate_by_hour(demo_df)
                fig_hour = px.line(
                    hourly_df,
                    x="hour",
                    y="fraud_rate_pct",
                    markers=True,
                    title="Fraud Percentage by Hour of Day",
                    labels={"hour": "Hour of Day (0-23)", "fraud_rate_pct": "Fraud Rate (%)"},
                )
                fig_hour.update_traces(line_color="#E4572E")
                st.plotly_chart(fig_hour, use_container_width=True)

            with tab3:
                st.subheader("Pearson Correlation with Fraud (Class)")
                corr_df = correlation_with_class(demo_df)
                st.dataframe(corr_df.head(15), use_container_width=True)

    # -------------------------------------------------------------
    # Page 7: About & Methodology
    # -------------------------------------------------------------
    elif page == "ℹ️ About & Methodology":
        st.title("ℹ️ About FraudSense: Methodology & Design")

        st.markdown(
            r"""
            ### 🎓 Project Background
            **FraudSense** is a 7th-semester Machine Learning laboratory project exploring 
            **Anomaly Detection for Extreme Class Imbalance** in credit card fraud detection.
            
            ---
            
            ### 📜 Core Engineering Principles
            1. **Strictly Pure NumPy**:
               - Every ML algorithm, metric (ROC-AUC, PR-AUC, F1), and statistical transformer 
                 is implemented entirely from scratch without scikit-learn, TensorFlow, or PyTorch.
            2. **No Data Leakage**:
               - Splitting is performed *before* preprocessing. Preprocessing statistics (means, standard deviations)
                 are learned **strictly on normal ($Class=0$) training transactions**.
            3. **Safe Serialization**:
               - Models and artifacts are serialized strictly via **JSON + NPZ** (`allow_pickle=False`). 
                 Python `pickle` is banned to guarantee security and cross-platform portability.
            4. **Startup Independence**:
               - The Streamlit web dashboard never trains models at startup; it only loads pre-computed, verified artifacts.
            
            ---
            
            ### ⚖️ Honest Limitations & Design Trade-Offs
            - **Anonymized Features:** Features $V_1 \dots V_{28}$ are anonymized PCA components. While this protects 
              cardholder privacy, it limits explainability to mathematical deviations rather than semantic fraud reasons 
              (e.g., merchant category or billing zip code mismatch).
            - **Decision Support, Not a Verdict:** ML models generate probabilistic indicators. All high-risk alerts 
              require human review before definitive cardholder action is taken.
            - **Concept Drift:** Anomaly detection assumes normal behavior remains stationary. Real-world fraud patterns 
              require recurring model re-calibration over time.
            """
        )


if __name__ == "__main__":
    main()
