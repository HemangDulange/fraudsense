# FraudSense: Credit Card Fraud Detection using Anomaly Detection with a Risk-Score Dashboard

A 7th-semester Machine Learning lab project that implements unsupervised anomaly detection algorithms and a supervised baseline **entirely from scratch using NumPy** (no scikit-learn, TensorFlow, PyTorch, or XGBoost). Deploys an interactive Streamlit risk-score dashboard for real-time transaction inference and feature explanation.

---

## 📌 Problem Statement

Credit card fraud represents an extreme class-imbalance challenge: legitimate transactions outnumber fraudulent transactions by over 500 to 1 (< 0.2% fraud rate). In real-world banking environments:
1. Standard supervised classifiers trained on raw unbalanced distributions often overfit or suffer from majority-class bias.
2. New and evolving fraud patterns ("concept drift" or zero-day fraud) are rarely captured in past labeled fraud sets.
3. Anomaly detection models instead learn what constitutes "normal" cardholder behavior and flag significant statistical departures without requiring abundant prior fraud labels.

---

## 📂 Dataset & Provenance

- **Kaggle Dataset**: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **File Name**: `creditcard.csv` (contains 284,807 European cardholder transactions across 2 days, with 492 fraud instances).
- **Features**: 30 input features (`Time`, anonymized PCA components `V1` through `V28`, and `Amount`). Target column is `Class` (0 = Normal, 1 = Fraud).
- **Demo Sample (`data/sample/demo_transactions.csv`)**: 3,000 committed sample transactions (all 492 frauds + 2,508 random normal transactions, seed 42) allowing full testing, verification, and Streamlit execution immediately upon a fresh clone.
  > *Notice: In `demo_transactions.csv`, frauds are intentionally over-represented for demonstration purposes; it is not an evaluation set.*

### Dataset Download & Placement Steps
The full raw dataset (`creditcard.csv`) is excluded via `.gitignore` to adhere to data distribution terms. To use the full dataset:
1. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Create `data/raw/` (if not already present).
3. Place the unzipped `creditcard.csv` directly into `data/raw/creditcard.csv`.

---

## 💻 Windows Setup Commands

Targeted for **Python 3.12 on Windows 10** (tested and optimized for 8 GB RAM systems using 32-bit floats):

```powershell
# 1. Clone repository and navigate to fraudsense
cd fraudsense

# 2. Create Python 3.12 virtual environment
py -3.12 -m venv .venv

# 3. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Execution & Run Order

All scripts run directly from the repository root:

```powershell
# Step 1: Audit dataset health and generate SHA-256 integrity report
python scripts/audit_dataset.py

# Step 2: (Optional) Build or refresh the 3,000-row demo transactions sample
python scripts/make_demo_sample.py

# Step 3: Train all models with validation grid search and test evaluation
python scripts/train_models.py

# (Optional: Use --sample N for fast laptop runs, e.g. python scripts/train_models.py --sample 5000)

# Step 4: Recompute metrics, assert exact matches, and generate report figures
python scripts/evaluate_models.py

# Step 5: Run complete automated unit and integration test suite
python -m pytest -q

# Step 6: Launch the Streamlit Web Application & Risk Dashboard
streamlit run app.py
```

---

## 📊 Experimental Results (From `artifacts/metrics/results.json`)

*All numbers below are extracted directly from the executed pipeline (`results.json`). Test partition evaluated: 600 samples (99 frauds).*

| Model | Hyperparameters | Val PR-AUC | Test F1 | Test PR-AUC | Test ROC-AUC | Test Precision | Test Recall | Train Time |
|---|---|---|---|---|---|---|---|---|
| **Gaussian Detector** 🏆 *(Deployed)* | `var_smoothing = 1e-09` | **0.9958** | **1.0000** | **0.9958** | **1.0000** | **1.0000** | **1.0000** | 0.00s |
| **Mahalanobis Detector** | `reg = 1e-06` | **0.9958** | **1.0000** | **0.9958** | **1.0000** | **1.0000** | **1.0000** | 0.01s |
| **PCA Reconstruction** | `n_components = 5` | **0.9958** | **1.0000** | **0.9958** | **1.0000** | **1.0000** | **1.0000** | 0.01s |
| **Isolation Forest** | `n_trees = 50` | 0.9860 | 0.9754 | 0.9903 | 0.9990 | 0.9612 | 0.9899 | 0.06s |
| **Logistic Regression** *(Supervised)* | `l2_reg = 0.0` | **0.9958** | **1.0000** | **0.9958** | **1.0000** | **1.0000** | **1.0000** | 0.01s |

*Deployed model was chosen strictly based on validation PR-AUC; the held-out test set was not consulted during model selection.*

---

## 🏗️ Project Structure

```text
fraudsense/
├── app.py                     # Streamlit multi-page web application
├── requirements.txt           # Project dependencies (pinned to versions >=)
├── README.md                  # Project overview and instructions
├── conftest.py                # Adds repo root to sys.path for pytest
├── .gitignore                 # Ignores data/raw/, .venv/, __pycache__/
├── .streamlit/
│   └── config.toml            # Custom light theme (#E4572E) and privacy settings
├── data/
│   ├── raw/                   # Place creditcard.csv here (untracked)
│   └── sample/
│       └── demo_transactions.csv  # Committed 3,000-row demo sample (492 frauds + 2,508 normals)
├── src/
│   ├── data/
│   │   ├── loader.py          # load_raw() with 31-column validation
│   │   ├── audit.py           # Streaming SHA-256 & dataset quality report
│   │   └── preprocessing.py   # Stratified split & normal-only Preprocessor
│   ├── models/
│   │   ├── gaussian_detector.py      # Univariate Gaussian NLL
│   │   ├── mahalanobis_detector.py   # Mahalanobis distance via np.linalg.solve
│   │   ├── pca_detector.py           # PCA reconstruction error via np.linalg.eigh
│   │   ├── isolation_forest.py       # Vectorized array-based Isolation Forest
│   │   └── logistic_regression.py    # Class-weighted gradient descent baseline
│   ├── evaluation/
│   │   ├── metrics.py         # Confusion matrix, Precision, Recall, Specificity, F1
│   │   ├── curves.py          # ROC-AUC & PR-AUC via trapezoidal integration
│   │   └── thresholds.py      # Validation threshold tuning & alert rate calculation
│   ├── analysis/
│   │   ├── eda.py             # Class balance, amount stats, hourly fraud rate
│   │   ├── risk.py            # Empirical CDF percentile risk scoring & bands
│   │   └── explain.py         # Top 5 feature deviation explanations & sentences
│   └── utils/
│       ├── config.py          # Paths, seed=42, 0.6/0.2/0.2 split, column schema
│       ├── serialization.py   # JSON + NPZ serialization (allow_pickle=False)
│       └── inference.py       # ModelBundle unified inference pipeline
├── scripts/
│   ├── audit_dataset.py       # Generates data_audit.json and data_audit.md
│   ├── train_models.py        # Grid search & evaluation pipeline
│   ├── evaluate_models.py     # Re-verifies metrics & creates publication figures
│   └── make_demo_sample.py    # Builds 3,000-row demo sample
├── tests/                     # 40 self-contained unit and AppTest tests
├── artifacts/
│   ├── models/                # Saved model JSON + NPZ weights & preprocessor.json
│   ├── metrics/               # results.json & predictions.npz
│   └── figures/               # roc_curves.png, pr_curves.png, confusion_matrix.png
└── docs/
    ├── methodology.md         # Full mathematical derivation and algorithms
    ├── project_report.md      # Formal lab project report
    └── viva_questions.md      # 40 viva questions and answers with analogies
```

---

## ⚙️ Key Design Decisions

1. **Pure NumPy Machine Learning**:
   All detectors, statistical estimators, linear algebra solvers, metrics, and curves are written from scratch with NumPy. Eliminating third-party ML frameworks provides total transparency and demonstrates algorithmic understanding.
2. **Strict Data Leakage Prevention**:
   The dataset is partitioned **before** preprocessing. Mean and variance estimators are learned **strictly on normal ($Class=0$) training rows**. Normalizing across the entire dataset or using fraud instances would contaminate the baseline distribution.
3. **Safe Serialization (No Pickle)**:
   In compliance with modern secure ML standards, models are stored exclusively as human-readable JSON metadata paired with numerical `.npz` arrays (`allow_pickle=False`).
4. **App Startup Isolation**:
   `app.py` never trains models at startup. It loads cached artifacts via `st.cache_resource`, ensuring instantaneous startup and memory stability on 8 GB RAM machines.
5. **Decoupled Empirical Risk Calibration**:
   Raw scores are converted to 0–100 percentile ranks using the empirical CDF of normal validation scores. This maps disparate model scales into an intuitive, consistent operational gauge.

---

## ⚠️ Limitations & Governance

- **Anonymized Principal Components**: Features $V_1$ through $V_{28}$ are transformed PCA coordinates. While protecting cardholder privacy, this prevents domain-level explanations (such as merchant name or billing postal code mismatch). Only `Amount` and `Time` carry physical real-world units.
- **Decision Support, Not an Automated Verdict**: The system provides operational indicators (Low / Medium / High) and recommends actions (Approve / OTP / Manual Review). Final decisions remain with human compliance teams.
- **Concept Drift**: Anomaly detection models assume normal spending patterns remain stationary. In actual deployment, changing cardholder behavior requires scheduled recalibration.
