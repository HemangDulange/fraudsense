# FraudSense: Technical Methodology & Mathematical Formulations

This document provides mathematical formulations, algorithmic derivations, and engineering specifications for the machine learning modules in **FraudSense**.

---

## 1. Data Transformation & Feature Engineering

### 1.1 Column Alignment & Dimensionality
The input transaction space $\mathbf{x} \in \mathbb{R}^{30}$ is defined in fixed canonical order:
$$\mathbf{x} = [\text{Time}, V_1, V_2, \dots, V_{28}, \text{Amount}]$$

### 1.2 Non-Linear Transformations
1. **Transaction Amount Compression**:
   Transaction amounts exhibit extreme positive skewness spanning multiple orders of magnitude. We apply the natural logarithmic transformation:
   $$\text{Amount}' = \ln(1 + \max(0, \text{Amount}))$$
   This compresses extreme monetary spikes and brings the marginal distribution closer to normality.

2. **Time to Cyclical Hour of Day**:
   The raw dataset measures `Time` as continuous seconds elapsed from the start of logging. To capture diurnal spending cycles, time is mapped to a 24-hour continuum:
   $$\text{Hour} = \frac{\text{Time} \pmod{86400}}{3600} \in [0.0, 24.0)$$

### 1.3 Normal-Only Baseline Normalization (Preventing Leakage)
To establish an unpolluted baseline of legitimate spending, parameters are estimated **strictly from normal ($Class = 0$) training samples**:
$$\mu_j = \frac{1}{N_{\text{norm}}} \sum_{i \in \text{Normal}} x_{ij}, \quad \sigma_j = \sqrt{\frac{1}{N_{\text{norm}}} \sum_{i \in \text{Normal}} (x_{ij} - \mu_j)^2}$$

If any feature exhibits near-zero dispersion ($\sigma_j < 10^{-7}$), $\sigma_j$ is set to $1.0$ to prevent numerical instability. Standardized scores are clipped to a bounded range:
$$z_j = \text{clip}\left(\frac{x_j - \mu_j}{\sigma_j}, -10.0, 10.0\right)$$

---

## 2. Anomaly Detection Algorithms

### 2.1 Univariate Gaussian Detector
Assumes features are conditionally independent under the normal class:
$$p(\mathbf{x}) = \prod_{j=1}^d \frac{1}{\sqrt{2\pi(\sigma_j^2 + \epsilon)}} \exp\left(-\frac{(x_j - \mu_j)^2}{2(\sigma_j^2 + \epsilon)}\right)$$

Taking the negative log-likelihood (NLL) avoids floating-point underflow:
$$\text{Score}_{\text{Gauss}}(\mathbf{x}) = -\ln p(\mathbf{x}) = \frac{d}{2}\ln(2\pi) + \frac{1}{2}\sum_{j=1}^d \ln(\sigma_j^2 + \epsilon) + \frac{1}{2}\sum_{j=1}^d \frac{(x_j - \mu_j)^2}{\sigma_j^2 + \epsilon}$$
where $\epsilon \in \{10^{-9}, 10^{-6}, 10^{-3}\}$ is the variance smoothing hyperparameter.

---

### 2.2 Multivariate Mahalanobis Distance Detector
Accounts for linear covariance among features. The empirical sample covariance matrix $\boldsymbol{\Sigma} \in \mathbb{R}^{d \times d}$ is computed on centered normal training data $\mathbf{X}_c = \mathbf{X}_{\text{norm}} - \boldsymbol{\mu}$:
$$\boldsymbol{\Sigma} = \frac{1}{N_{\text{norm}} - 1} \mathbf{X}_c^T \mathbf{X}_c$$

To guarantee positive-definiteness and prevent non-invertibility from collinear features, Tikhonov (ridge) regularization is applied:
$$\boldsymbol{\Sigma}_{\text{reg}} = \boldsymbol{\Sigma} + \lambda \mathbf{I}_d, \quad \lambda \in \{10^{-6}, 10^{-4}, 10^{-2}\}$$

The squared Mahalanobis distance is:
$$D_M^2(\mathbf{x}) = (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}_{\text{reg}}^{-1} (\mathbf{x} - \boldsymbol{\mu})$$

**Numerical Stability via `np.linalg.solve`:** Rather than explicitly computing matrix inversion $\boldsymbol{\Sigma}_{\text{reg}}^{-1}$ (which is numerically volatile), we solve the linear system:
$$\boldsymbol{\Sigma}_{\text{reg}} \mathbf{z} = (\mathbf{x} - \boldsymbol{\mu}) \implies \text{Score}_{\text{Maha}}(\mathbf{x}) = (\mathbf{x} - \boldsymbol{\mu})^T \mathbf{z}$$

---

### 2.3 Principal Component Analysis (PCA) Reconstruction Error
Computes eigenvectors of the covariance matrix using symmetric eigendecomposition via `np.linalg.eigh`:
$$\boldsymbol{\Sigma} \mathbf{v}_k = \lambda_k \mathbf{v}_k, \quad \lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_d$$

We construct the projection matrix $\mathbf{V}_k = [\mathbf{v}_1, \dots, \mathbf{v}_k] \in \mathbb{R}^{d \times k}$ containing the top $k$ principal components ($k \in \{5, 10, 15, 20\}$).

For transaction $\mathbf{x}$, the projection into the $k$-dimensional subspace and subsequent reconstruction back into original coordinates is:
$$\hat{\mathbf{x}}_c = (\mathbf{x} - \boldsymbol{\mu}) \mathbf{V}_k \mathbf{V}_k^T$$

The anomaly score is the squared Euclidean reconstruction error:
$$\text{Score}_{\text{PCA}}(\mathbf{x}) = \|\mathbf{x}_c - \hat{\mathbf{x}}_c\|_2^2 = \sum_{j=1}^d \left( (x_j - \mu_j) - \hat{x}_{c, j} \right)^2$$

Normal transactions reconstruct with negligible loss; anomalous transactions violate correlation patterns and produce large reconstruction residuals.

---

### 2.4 Isolation Forest (From Scratch)
Isolates anomalies by recursive binary partitioning. Since anomalies are sparse and atypical, they isolate in fewer random splits than normal points.

1. **Subsampling**: For each tree $t \in \{1, \dots, T\}$, sample $\psi = 256$ points without replacement.
2. **Height Limit**: $h_{\max} = \lceil \log_2(\psi) \rceil = \lceil \log_2(256) \rceil = 8$.
3. **Node Partitioning**: Randomly select feature $j \in \{1, \dots, d\}$ and draw split threshold $\theta \sim \text{Uniform}(\min(\mathbf{x}_{:, j}), \max(\mathbf{x}_{:, j}))$.
4. **Average Path Length Normalization**:
   The average path length of an unsuccessful search in a Binary Search Tree (BST) of size $n$ is:
   $$c(n) = \begin{cases} 2\left(\ln(n - 1) + 0.5772156649\right) - \frac{2(n - 1)}{n} & \text{if } n > 2 \\ 1.0 & \text{if } n = 2 \\ 0.0 & \text{if } n \le 1 \end{cases}$$
5. **Anomaly Score**:
   $$\text{Score}_{\text{iForest}}(\mathbf{x}) = 2^{-\frac{\mathbb{E}[h(\mathbf{x})]}{c(\psi)}}$$
   where $\mathbb{E}[h(\mathbf{x})]$ is the mean path length across all trees. If $\mathbb{E}[h(\mathbf{x})] \to 0$, score $\to 1$ (highly anomalous).

---

### 2.5 Supervised Baseline: Logistic Regression
Serves as an empirical baseline. Incorporates balanced class weighting to handle severe imbalance:
$$w_0 = \frac{N}{2 N_0}, \quad w_1 = \frac{N}{2 N_1}$$

Weighted Binary Cross-Entropy loss with L2 regularization:
$$\mathcal{L}(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^N s_i \left[ y_i \ln(\sigma(z_i) + \epsilon) + (1 - y_i) \ln(1 - \sigma(z_i) + \epsilon) \right] + \frac{\lambda}{2} \|\mathbf{w}\|_2^2$$
where $z_i = \mathbf{x}_i^T \mathbf{w} + b$, $\sigma(z) = \frac{1}{1 + e^{-\text{clip}(z, -25, 25)}}$, and $s_i = w_1 y_i + w_0 (1 - y_i)$.

Gradients evaluated via batch gradient descent:
$$\nabla_{\mathbf{w}} = \frac{1}{N} \mathbf{X}^T \mathbf{g} + \lambda \mathbf{w}, \quad \nabla_b = \frac{1}{N} \sum_{i=1}^N g_i, \quad g_i = s_i (\sigma(z_i) - y_i)$$

---

## 3. Evaluation & Metrics Formulation

### 3.1 Confusion Matrix & Thresholding
Given decision threshold $\tau$, predicted binary labels are $\hat{y}_i = \mathbb{I}(\text{Score}(\mathbf{x}_i) \ge \tau)$:
- $\text{Precision} = \frac{TP}{TP + FP}$
- $\text{Recall (TPR)} = \frac{TP}{TP + FN}$
- $\text{Specificity (TNR)} = \frac{TN}{TN + FP}$
- $\text{FPR} = \frac{FP}{FP + TN} = 1 - \text{Specificity}$
- $F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
- $\text{Balanced Accuracy} = \frac{\text{Recall} + \text{Specificity}}{2}$

### 3.2 Area Under Curve (Trapezoidal Rule)
Numerical integration over $M$ sorted evaluation points:
$$\text{AUC} = \sum_{i=1}^{M-1} (x_{i+1} - x_i) \cdot \frac{y_{i+1} + y_i}{2}$$
- **ROC-AUC**: Integrated with $x = \text{FPR}$, $y = \text{TPR}$.
- **PR-AUC**: Integrated with $x = \text{Recall}$, $y = \text{Precision}$. Under extreme class imbalance, PR-AUC provides a much sharper metric than ROC-AUC because it is sensitive to the true fraud detection rate without being artificially inflated by vast quantities of true negatives.

---

## 4. Operational Risk Scoring & Explainability

### 4.1 Empirical CDF Percentile Ranking
Given validation normal scores $\mathcal{S}_{\text{norm}} = \{s_1 \le s_2 \le \dots \le s_M\}$:
$$\text{Risk Percentile}(s) = \frac{\sum_{i=1}^M \mathbb{I}(s_i \le s)}{M} \times 100\%$$

- **Low Risk** ($< 90\%$): Approve.
- **Medium Risk** ($90\%$ to Deployed Threshold $\tau$): Step-up OTP Verification.
- **High Risk** ($\ge \tau$): Hold and route to human fraud review team.

### 4.2 Top 5 Feature Attribution
Total transaction abnormality deviation:
$$D_{\text{total}} = \sum_{j=1}^{30} |z_j|$$
For each feature $j$, its proportional deviation contribution is:
$$\text{Share}_j = \frac{|z_j|}{D_{\text{total}}} \times 100\%$$
Features are ranked in descending order of $|z_j|$ to surface the top 5 drivers behind an anomaly.
