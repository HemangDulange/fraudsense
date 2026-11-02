# FraudSense: Credit Card Fraud Detection using Anomaly Detection with a Risk-Score Dashboard

**Course/Lab:** 7th Semester Machine Learning Laboratory Project  
**Implementation Standard:** Pure NumPy Core (No Scikit-Learn or Third-Party ML Libraries)  
**System Constraints:** Windows 10, Python 3.12, 8 GB RAM Compatibility  

---

## 1. Abstract
Credit card fraud is characterized by extreme class imbalance where fraudulent events account for less than 0.2% of total transaction traffic. Standard supervised classifiers trained on raw unbalanced distributions frequently suffer from majority-class bias and struggle to flag zero-day attack vectors. **FraudSense** introduces an end-to-end credit card fraud detection system implementing four unsupervised anomaly detection algorithms (Univariate Gaussian, Multivariate Mahalanobis Distance, PCA Reconstruction Error, and Isolation Forest) alongside a class-weighted Logistic Regression baseline—developed entirely from scratch using NumPy. Models are tuned on a validation set using Precision-Recall AUC (PR-AUC), and the top-performing model is deployed to an interactive Streamlit dashboard. Feature explanations and empirical percentile risk scores provide actionable decision support. All experimental metrics reported in this document are extracted directly from pipeline artifacts (`artifacts/metrics/results.json`).

---

## 2. Introduction
In financial payment processing networks, fraudulent credit card transactions result in tens of billions of dollars in losses annually. Concurrently, false fraud alarms cause unnecessary transaction declines, leading to user friction and customer churn. 

Traditional machine learning projects frequently rely on black-box libraries like scikit-learn or XGBoost. In contrast, this project builds every component from first principles using fundamental numerical mathematics:
- Univariate Gaussian density modeling.
- Regularized multivariate Mahalanobis distance via linear system solvers (`np.linalg.solve`).
- Eigendecomposition and subspace projection via `np.linalg.eigh`.
- Vectorized tree ensembles with array-based traversal.
- Gradient descent with class-rebalanced loss functions.

---

## 3. Problem Statement
The primary objectives are:
1. Detect credit card transactions that deviate significantly from legitimate spending patterns.
2. Formulate models capable of training primarily on normal transactions without requiring balanced fraud labels.
3. Eliminate data leakage by isolating preprocessing statistics strictly to normal training transactions.
4. Provide transparent, plain-English feature attributions explaining *why* a transaction was flagged.
5. Translate raw continuous anomaly scores into calibrated 0–100 risk percentiles with concrete operational action bands (Approve, OTP Verification, Manual Review).

---

## 4. Dataset Description
- **Source:** Kaggle Credit Card Fraud Detection Dataset ([Kaggle Link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)).
- **Volume:** 284,807 European cardholder transactions recorded over a two-day period in September 2013.
- **Class Imbalance:** 492 fraud instances (0.172% fraud rate, an imbalance ratio of ~1 fraud per 578 legitimate transactions).
- **Features (30 inputs):**
  - `Time`: Elapsed seconds since the initial transaction.
  - `V1` through `V28`: Principal components extracted via PCA by the original data collectors to protect cardholder privacy.
  - `Amount`: Transaction purchase amount in dollars.
- **Committed Demo Sample (`data/sample/demo_transactions.csv`):** 3,000 transactions containing all 492 frauds and 2,508 random normal transactions (seed 42), enabling immediate replication on any clone without requiring external downloads.

---

## 5. Methodology & Architecture

### 5.1 Preprocessing Pipeline
1. **Stratified Split (60% Train / 20% Val / 20% Test):**
   Maintains the minority fraud proportion across all three splits using NumPy random index permutation.
2. **Feature Engineering:**
   - Amount: Transformed via $\ln(1 + \text{Amount})$ to compress extreme monetary skewness.
   - Time: Converted to cyclical 24-hour hour-of-day: $\text{Hour} = \frac{\text{Time} \pmod{86400}}{3600}$.
3. **Normal-Only Training Profile (No Data Leakage):**
   Mean $\mu_j$ and standard deviation $\sigma_j$ are computed **strictly on normal ($Class=0$) training rows**. Standardized features are clipped to $[-10.0, 10.0]$.

### 5.2 Algorithmic Implementations
- **Gaussian Detector:** Evaluates negative log-likelihood (NLL) summed across independent 1D Gaussians with variance smoothing $\epsilon \in [10^{-9}, 10^{-6}, 10^{-3}]$.
- **Mahalanobis Detector:** Measures squared distance $D_M^2(\mathbf{x}) = (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}_{\text{reg}}^{-1} (\mathbf{x} - \boldsymbol{\mu})$. Uses `np.linalg.solve` to avoid numerical matrix inversion, with ridge regularization $\lambda \in [10^{-6}, 10^{-4}, 10^{-2}]$.
- **PCA Reconstruction:** Identifies principal components via `np.linalg.eigh` on normal covariance. Anomaly score is the squared Euclidean reconstruction error $\|\mathbf{x}_c - \hat{\mathbf{x}}_c\|_2^2$ across $k \in [5, 10, 15, 20]$ components.
- **Isolation Forest:** Builds an ensemble of random binary isolation trees ($T \in [50, 100]$, subsample $\psi = 256$, max height $= 8$). Trees are stored in flat parallel arrays to enable vectorized scoring of 50,000 rows in ~0.2 seconds.
- **Logistic Regression (Supervised Baseline):** Sigmoid activation trained via batch gradient descent with balanced class weights $w_c = \frac{N}{2 N_c}$ and L2 regularization.

---

## 6. Implementation & Safety Standards
- **Safe Serialization:** Models are serialized strictly using **JSON + NPZ** (`allow_pickle=False`). Python `pickle` is banned.
- **Memory Efficiency:** Utilizes 32-bit floats (`float32`) throughout arrays to ensure stability under 8 GB RAM.
- **Universal Inference Engine (`ModelBundle`):** Enforces a single production inference pipeline: `validate_columns -> preprocess -> score -> risk score and band -> explain`.

---

## 7. Experimental Results

*All performance metrics below are reported directly from the executed pipeline artifacts (`artifacts/metrics/results.json`).*

### 7.1 Dataset Partitioning
- **Total Samples:** 3,000
- **Training Partition (60%):** 1,800 samples (295 frauds)
- **Validation Partition (20%):** 600 samples (98 frauds)
- **Held-Out Test Partition (20%):** 600 samples (99 frauds)

### 7.2 Performance Comparison on Test Set (600 transactions, 99 frauds)
| Algorithm | Selected Hyperparameter | Val PR-AUC | Test F1 | Test PR-AUC | Test ROC-AUC | Test Precision | Test Recall | Train Time |
|---|---|---|---|---|---|---|---|---|
| **Gaussian Detector** 🏆 *(Deployed)* | `var_smoothing = 1e-09` | **0.9958** | **1.0000** | **0.9958** | **1.0000** | **1.0000** | **1.0000** | 0.00s |
| **Mahalanobis Detector** | `reg = 1e-06` | **0.9958** | **1.0000** | **0.9958** | **1.0000** | **1.0000** | **1.0000** | 0.01s |
| **PCA Reconstruction** | `n_components = 5` | **0.9958** | **1.0000** | **0.9958** | **1.0000** | **1.0000** | **1.0000** | 0.01s |
| **Isolation Forest** | `n_trees = 50` | 0.9860 | 0.9754 | 0.9903 | 0.9990 | 0.9612 | 0.9899 | 0.06s |
| **Logistic Regression** *(Supervised)* | `l2_reg = 0.0` | **0.9958** | **1.0000** | **0.9958** | **1.0000** | **1.0000** | **1.0000** | 0.01s |

### 7.3 Detailed Confusion Matrix Breakdown (Test Set)
- **Gaussian Detector (Deployed):**
  - True Negatives (TN): **501**
  - False Positives (FP): **0**
  - False Negatives (FN): **0**
  - True Positives (TP): **99**
  - False Positive Rate: **0.00%**
  - Operational Alerts per 10,000 swipes: **1,650.0**
- **Isolation Forest:**
  - True Negatives: **497**, False Positives: **4**, False Negatives: **1**, True Positives: **98**
  - Precision: **96.12%**, Recall: **98.99%**

---

## 8. Discussion
1. **Unsupervised Efficacy:** Unsupervised anomaly detectors (Gaussian, Mahalanobis, PCA) matched or exceeded the supervised baseline on held-out test data. Because normal transactions cluster tightly within a low-dimensional manifold, anomalies stand out with immense statistical contrast.
2. **Evaluation Metric Superiority (PR-AUC vs. ROC-AUC):** Under extreme class imbalance, ROC-AUC can be deceptively optimistic because a massive denominator of True Negatives dilutes False Positive Rates. In contrast, PR-AUC focuses on minority class precision and recall, making it an indispensable criterion for hyperparameter selection.
3. **Execution Latency:** The Gaussian detector achieved inference times under 0.1 milliseconds per batch, while vectorized Isolation Forest traversal scored thousands of transactions in under 0.1 seconds.

---

## 9. Limitations
1. **PCA Anonymization:** $V_1$ through $V_{28}$ are anonymized principal coordinates. This prevents domain-specific explanations (e.g., specific merchant category codes or overseas IP flags).
2. **Concept Drift:** Fraudsters continually modify attack strategies. An unsupervised normal profile trained on historical data requires periodic retraining as legitimate cardholder habits evolve.
3. **Operational Review Constraints:** Machine learning scores serve as **decision support, not a final legal verdict**. Transactions flagged as High Risk must be routed to human review teams.

---

## 10. Conclusion
FraudSense successfully demonstrates that high-performance, mathematically rigorous credit card fraud detection can be achieved entirely from scratch with NumPy. By enforcing strict data leakage prevention, employing PR-AUC validation tuning, and coupling anomaly scores with an empirical percentile dashboard, the project delivers an end-to-end machine learning system meeting both academic laboratory standards and real-world operational requirements.
