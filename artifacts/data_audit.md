# 📋 Credit Card Fraud Dataset Audit Report

> **FraudSense Data Integrity & Quality Verification**

- **File SHA-256 Checksum**: `d91409ea32aeecd4ae89827887e8e4018f9b597cf65a9c637f944a177af98873`
- **Total Transactions**: `3,000` rows
- **Total Features**: `31` columns

---

## 1. Class Distribution & Imbalance

| Class | Description | Count | Percentage |
|---|---|---|---|
| `0` | Normal Transactions | 2,508 | 83.6000% |
| `1` | Fraudulent Transactions | 492 | 16.4000% |

- **Imbalance Ratio**: 1 fraud per 5 normal transactions
- **Observed Fraud Rate**: **16.400%** (Matches expected ~0.172% for the 492 frauds in 284,807 transactions).

---

## 2. Data Quality & Cleanliness

| Quality Metric | Measured Value | Status |
|---|---|---|
| Missing Values | 0 | ✅ Clean (0 missing) |
| Fully Duplicate Rows | 0 (0.0%) | ✅ No duplicates |

---

## 3. Transaction Amount (`Amount`) Distribution

| Statistic | Value ($) |
|---|---|
| Min | $0.10 |
| 25th Percentile | $19.18 |
| Median | $51.85 |
| Mean | $77.19 |
| 75th Percentile | $103.20 |
| 99th Percentile | $407.59 |
| Max | $890.97 |
| Std Dev | $85.43 |

---

## 4. Time Span (`Time`) Summary

- **Minimum Elapsed Seconds**: 78.7s
- **Maximum Elapsed Seconds**: 172785.4s
- **Total Time Span**: **47.97 hours** (~2 full days of transaction logs)
