# FraudSense Viva Voce Guide: 40 Comprehensive Questions & Answers

This guide prepares students for lab examinations and project defense. Every answer is written in plain English with simple everyday analogies.

---

## Category 1: Anomaly Detection & Problem Formulation

### Q1: What is the core difference between anomaly detection and standard supervised classification?
**Answer:** Supervised classification trains a model by showing it both normal and fraudulent examples so it can learn a boundary between them. Anomaly detection only needs to see what *normal* behavior looks like; anything that departs significantly from that normal pattern is flagged as suspicious.  
*Everyday Analogy:* Imagine an art authenticator who studies Leonardo da Vinci's brushstrokes for 20 years. When a fake painting arrives, they don't need to have seen that specific forger's work before—they simply recognize that the brushstrokes do not match da Vinci's authentic style.

### Q2: Why is anomaly detection especially suited for credit card fraud?
**Answer:** In credit card processing, fraud represents less than 0.2% of transactions. Moreover, fraudsters constantly invent new tricks (zero-day fraud). Anomaly detection excels because it detects brand-new, unseen fraud patterns simply because they deviate from legitimate cardholder habits.

### Q3: What is "zero-day" fraud?
**Answer:** A fraud scheme that has never been observed in historical training data before. Supervised models often miss it because they look for known signatures; anomaly detection flags it because it looks abnormal.

### Q4: Why can't we just use unsupervised clustering like basic K-Means for fraud detection?
**Answer:** K-Means forces every point into a cluster center and struggles with high-dimensional noise. Anomaly detection specifically focuses on the thin density tails and sparse outskirts rather than grouping points into round blobs.

### Q5: What does "decision support, not a verdict" mean in financial fraud systems?
**Answer:** The machine learning model provides an estimated risk score (e.g., 95th percentile risk) and recommends an operational action, but it does not make a final legal verdict. High-risk transactions are paused for step-up authentication (OTP) or human review to avoid wrongly freezing a customer's card.

---

## Category 2: Class Imbalance & Why Accuracy Fails

### Q6: Why is Accuracy a completely misleading metric in fraud detection?
**Answer:** In a dataset with 99.8% normal transactions and 0.2% frauds, a dummy "lazy" model that labels every single swipe as "Normal" achieves 99.8% accuracy. Yet, it catches zero frauds and is totally useless.  
*Everyday Analogy:* A broken fire alarm that never sounds is 99.9% accurate on days with no fire, but completely fails its only job when an actual fire breaks out.

### Q7: What is the class imbalance ratio in credit card fraud detection?
**Answer:** Typically around 1 fraud per 500 to 1,000 normal transactions (< 0.2%). In the benchmark European cardholder dataset, there are only 492 frauds out of 284,807 transactions (~0.172%).

### Q8: What happens if you train an unweighted supervised model on severe class imbalance?
**Answer:** The optimization loss function is dominated by the millions of normal examples. The model minimizes its loss by ignoring the rare fraud class entirely and predicting "normal" for almost everything.

### Q9: What is Balanced Accuracy, and how does it fix standard accuracy?
**Answer:** Balanced Accuracy is the arithmetic mean of Recall on the fraud class and Specificity on the normal class: $\frac{\text{Recall} + \text{Specificity}}{2}$. If a model predicts all zeros, its recall is 0 and specificity is 1, yielding a balanced accuracy of 50% (equivalent to a random coin flip).

### Q10: Why do we use stratified splitting instead of random shuffling?
**Answer:** Random shuffling on tiny minority classes could accidentally place almost all frauds into the training set and leave zero frauds in the test set. Stratified splitting forces the exact same 0.17% fraud percentage into train, validation, and test sets.

---

## Category 3: Evaluation Metrics (PR-AUC vs. ROC-AUC)

### Q11: What is the difference between Precision and Recall?
**Answer:** 
- **Recall (Sensitivity):** Out of all the real frauds that happened, what percentage did our system catch?
- **Precision:** Out of all the alerts our system raised, what percentage were actual frauds (and not false alarms)?  
*Everyday Analogy:* A metal detector at an airport that beeps at belt buckles has high recall (it never misses a weapon) but low precision (lots of false alarms).

### Q12: Why is PR-AUC preferred over ROC-AUC under extreme class imbalance?
**Answer:** ROC-AUC plots True Positive Rate vs. False Positive Rate ($FPR = \frac{FP}{FP + TN}$). Because the number of True Negatives ($TN$) in banking is astronomical, the denominator is massive, making FPR look tiny even when hundreds of false alarms occur. PR-AUC ignores True Negatives entirely and evaluates Precision vs. Recall directly on the minority class.

### Q13: What does an ROC-AUC of 0.5 versus 1.0 mean?
**Answer:** An ROC-AUC of 0.5 represents a useless random guess (like flipping a coin). An ROC-AUC of 1.0 indicates a perfect model where every single fraud scores higher than every normal transaction.

### Q14: What is the baseline value for a Precision-Recall curve?
**Answer:** For an ROC curve, a random classifier produces a diagonal line at 0.5. For a PR curve, a random guess produces a horizontal line at the dataset's fraud rate (e.g., $y = 0.0017$ or 0.17%). Any curve significantly above this line demonstrates meaningful fraud predictive power.

### Q15: How did we compute ROC-AUC and PR-AUC without scikit-learn?
**Answer:** Using numerical trapezoidal integration from scratch with NumPy. We sort the curve coordinates $(x_i, y_i)$ and compute the area as the sum of trapezoids: $\sum (x_{i+1} - x_i) \frac{y_{i+1} + y_i}{2}$.

---

## Category 4: Data Leakage & Preprocessing Principles

### Q16: What is data leakage, and how could it occur during preprocessing?
**Answer:** Data leakage occurs when information from outside the training set leaks into the model during training. If we compute the mean, standard deviation, or log scaling across the *entire* dataset before splitting, the test set's distribution leaks into the training pipeline, producing artificially optimistic results that fail in production.

### Q17: Why do we fit the Preprocessor ONLY on normal ($Class = 0$) training rows?
**Answer:** Anomaly detection is designed to model the statistical parameters of *legitimate* transactions. If we included fraud transactions when calculating the mean and standard deviation, extreme fraudulent spending would pull the mean up and inflate the standard deviation, "normalizing" the fraud and making anomalies harder to detect.

### Q18: Why do we apply $\ln(1 + \text{Amount})$ to the transaction amount?
**Answer:** Credit card transaction amounts are severely skewed: 90% of transactions are small (under \$50), but a few are huge (\$10,000+). The logarithm compresses this heavy tail so extreme values don't overwhelm distance calculations, and adding 1 ensures that \$0 transactions map to $\ln(1) = 0$ instead of negative infinity.

### Q19: Why do we convert Time to cyclical hour of day: $(\text{Time} \pmod{86400}) / 3600$?
**Answer:** The raw data measures cumulative seconds over two days. But fraud risk is correlated with the time of day (e.g., fraudulent spending often peaks between 2:00 AM and 5:00 AM). Modulo 86,400 (seconds in 24 hours) converts seconds into a continuous 0-to-24 hour daily clock.

### Q20: Why do we clip standardized z-scores to $[-10, 10]$?
**Answer:** Extreme outlier values (e.g., a z-score of 500) can cause numerical overflow, gradient explosion, or matrix singularity in linear algebra routines (`np.linalg.solve`, `eigh`). Clipping ensures robust numerical stability without changing the rank order of extreme anomalies.

---

## Category 5: Thresholds & Operational Risk

### Q21: What is a decision threshold?
**Answer:** Anomaly detectors produce continuous scores (e.g., 42.8). The decision threshold is the cut-off value $\tau$: transactions scoring $\ge \tau$ are classified as fraud, while scores $< \tau$ are classified as normal.

### Q22: How did we select the optimal threshold in FraudSense?
**Answer:** We evaluated candidate thresholds strictly on the **validation set** to find the threshold that maximizes the **F1-score**. The held-out test set was never consulted during threshold tuning.

### Q23: What is the business trade-off when moving the threshold higher or lower?
**Answer:** 
- **Lowering threshold:** Catches more fraud (higher Recall) but increases false alarms (lower Precision), frustrating legitimate customers and overwhelming review teams.
- **Raising threshold:** Reduces false alarms (higher Precision) but misses more fraudulent transactions (lower Recall), leading to direct financial loss.

### Q24: What is the "Alerts per 10,000" metric?
**Answer:** It calculates the expected number of flagged transactions for every 10,000 card swipes: $\frac{\text{Flagged Transactions}}{\text{Total Transactions}} \times 10{,}000$. Bank fraud managers use this number to determine staffing requirements for manual investigation queues.

### Q25: How does FraudSense map raw scores into 0–100 percentile risk scores?
**Answer:** By using the Empirical Cumulative Distribution Function (CDF) of normal validation scores. If a raw score is higher than 95% of all legitimate transactions in validation, its Risk Score is 95%.

---

## Category 6: Univariate Gaussian Detector

### Q26: How does the Univariate Gaussian Anomaly Detector work?
**Answer:** It fits an independent 1D bell curve to each feature of normal transactions by computing its mean $\mu_j$ and variance $\sigma_j^2$. For a new transaction, it calculates the probability density under each feature's bell curve. Points with very low probability are flagged as anomalies.

### Q27: Why do we take the Negative Log-Likelihood (NLL) instead of multiplying probabilities?
**Answer:** Multiplying 30 tiny decimal probabilities (e.g., $0.001 \times 0.0004 \times \dots$) causes floating-point arithmetic underflow (the computer rounds to 0.0). Logarithms convert multiplication into addition: $-\ln(a \cdot b) = -\ln(a) - \ln(b)$. Summing negative logs keeps numbers positive, stable, and larger for rare events.

### Q28: What is the role of `var_smoothing`?
**Answer:** A small positive constant (e.g., $10^{-9}$) added to the variance: $\sigma_j^2 + \epsilon$. It prevents division by zero if a feature has near-zero variance.

---

## Category 7: Multivariate Mahalanobis Distance

### Q29: What is the difference between Euclidean Distance and Mahalanobis Distance?
**Answer:** Euclidean distance assumes all features are independent and have equal scale (measuring straight-line circular distance). Mahalanobis distance accounts for variance and feature correlations using the covariance matrix.  
*Everyday Analogy:* Height and shoe size are correlated. A 6'2" person with size 11 shoes has above-average numbers, but Euclidean distance would treat them as abnormal, whereas Mahalanobis distance recognizes that this combination is completely normal.

### Q30: What is the formula for squared Mahalanobis distance?
**Answer:** $D_M^2(\mathbf{x}) = (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}_{\text{reg}}^{-1} (\mathbf{x} - \boldsymbol{\mu})$, where $\boldsymbol{\Sigma}_{\text{reg}} = \boldsymbol{\Sigma} + \lambda \mathbf{I}$ is the ridge-regularized covariance matrix.

### Q31: Why did we use `np.linalg.solve` instead of inverting the covariance matrix with `inv()`?
**Answer:** Explicit matrix inversion ($\boldsymbol{\Sigma}^{-1}$) is computationally slower and prone to severe numerical rounding errors. Solving the linear system $\boldsymbol{\Sigma} \mathbf{z} = (\mathbf{x} - \boldsymbol{\mu})$ directly for $\mathbf{z}$ is much more stable and accurate.

---

## Category 8: PCA Reconstruction Error

### Q32: How does PCA detect anomalies through reconstruction error?
**Answer:** PCA learns the $k$ directions of greatest variance from normal transactions. When a transaction is compressed into these $k$ components and then reconstructed back to 30 dimensions, normal transactions reconstruct almost perfectly. However, anomalies deviate along non-principal directions, producing a high reconstruction residual error $\|\mathbf{x} - \hat{\mathbf{x}}\|_2^2$.  
*Everyday Analogy:* Projecting a smooth vase onto a 2D wall allows you to rebuild its 3D shape easily. But if the vase has jagged metal spikes sticking out in odd angles, the 2D shadow loses them, and rebuilding the vase produces huge errors where the spikes were.

### Q33: How are principal components computed from scratch?
**Answer:** We compute the sample covariance matrix $\boldsymbol{\Sigma} = \frac{1}{N-1}\mathbf{X}_c^T \mathbf{X}_c$ and find its eigenvalues and eigenvectors using `np.linalg.eigh`. Sorting the eigenvectors in descending order of eigenvalue gives the principal directions of variance.

### Q34: What happens if $k$ (number of components) is too small or too large?
**Answer:** If $k$ is too small (e.g., 1), normal transactions lose too much detail and get high reconstruction errors. If $k$ is too large (close to 30), the model reconstructs anomalies as well, diminishing the error contrast. We tuned $k \in \{5, 10, 15, 20\}$ on validation PR-AUC.

---

## Category 9: Isolation Forest from Scratch

### Q35: How does Isolation Forest isolate anomalies?
**Answer:** It builds random binary decision trees. At each step, it picks a random feature and a random split value between that feature's min and max. Because anomalies have extreme values or exist in sparse space, they get isolated in very few splits (short path length). Normal points are clustered densely together and require many splits to isolate.  
*Everyday Analogy:* Playing "20 Questions". If someone in a room is wearing a neon astronaut suit, you isolate them in 1 question: "Are you wearing an astronaut suit?". But isolating someone in a business suit requires many detailed questions.

### Q36: What is the purpose of $c(n)$ in the Isolation Forest score?
**Answer:** $c(n)$ represents the average path length of an unsuccessful search in a Binary Search Tree of size $n$. Dividing the observed average path length by $c(n)$ normalizes the score across different sample sizes so scores fall between 0 and 1: $\text{Score} = 2^{-\frac{\mathbb{E}[h]}{c(n)}}$.

### Q37: How did we vectorize tree scoring to handle 50,000 rows in under 0.2 seconds?
**Answer:** Trees are stored as flat parallel NumPy arrays (`feature`, `threshold`, `left_child`, `right_child`, `node_size`). Instead of traversing rows one-by-one with recursive function calls, traversal is vectorized across all 50,000 rows simultaneously over at most 8 loop iterations (since maximum tree height $= \lceil\log_2(256)\rceil = 8$).

---

## Category 10: Supervised Baseline & Safe Serialization

### Q38: How does the class-weighted Logistic Regression baseline work?
**Answer:** It fits a linear boundary $z = \mathbf{x}^T \mathbf{w} + b$ passed through a sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$. To prevent majority-class bias, sample weights are inversely proportional to class frequencies: $w_0 = \frac{N}{2 N_0}$ and $w_1 = \frac{N}{2 N_1}$, penalizing mistakes on rare fraud cases much more heavily during gradient descent.

### Q39: Why is Python's `pickle` library banned in this project?
**Answer:** `pickle` has severe security vulnerabilities (it can execute arbitrary malicious code upon loading unverified bytes) and suffers from version incompatibility between Python releases. We serialized all models using **JSON** for human-readable parameters and compressed **NPZ** (`allow_pickle=False`) for numerical arrays.

### Q40: What happens if an unseen category or missing feature arrives at runtime?
**Answer:** The `ModelBundle` enforces strict column schema validation. If required features are missing, it raises an informative `ValueError` before inference. All numerical transforms apply fixed bounds and zero-variance replacement ($1.0$) to guarantee deterministic, crash-proof execution.
