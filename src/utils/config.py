"""Configuration module for FraudSense.

This module centralizes all file paths, global random seeds,
data split ratios, and column definitions used across the project.
All paths are defined relative to the project root directory.
"""

from pathlib import Path

# =====================================================================
# Project Directories & Paths
# =====================================================================
# The root folder of the project (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_SAMPLE_DIR = DATA_DIR / "sample"

# Raw dataset file path (from Kaggle Credit Card Fraud Detection)
RAW_DATA_FILE = DATA_RAW_DIR / "creditcard.csv"

# Demo transactions file path for Streamlit dashboard and tests
SAMPLE_DATA_FILE = DATA_SAMPLE_DIR / "demo_transactions.csv"

# Artifacts storage directories
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
FIGURES_DIR = ARTIFACTS_DIR / "figures"

# Documentation directory
DOCS_DIR = PROJECT_ROOT / "docs"

# =====================================================================
# Reproducibility & Train/Val/Test Split Ratios
# =====================================================================
# Master random seed for reproducible splits and initializations
RANDOM_SEED = 42

# Data split ratios: 60% train, 20% validation, 20% test
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# =====================================================================
# Dataset Columns Definition
# =====================================================================
# The Kaggle Credit Card dataset contains 31 columns:
# 1. 'Time': Seconds elapsed between this transaction and the first transaction
# 2. 'V1' through 'V28': Principal components obtained via PCA (anonymized)
# 3. 'Amount': Transaction amount in dollars
# 4. 'Class': Target label (1 = fraudulent transaction, 0 = normal transaction)

TIME_COL = "Time"
PCA_FEATURE_COLS = [f"V{i}" for i in range(1, 29)]
AMOUNT_COL = "Amount"
TARGET_COL = "Class"

# Feature columns (30 total input features)
FEATURE_COLS = [TIME_COL] + PCA_FEATURE_COLS + [AMOUNT_COL]

# All expected columns in the dataset (31 total)
ALL_COLS = FEATURE_COLS + [TARGET_COL]

# Total number of expected columns
NUM_EXPECTED_COLS = 31

# =====================================================================
# Memory Optimization
# =====================================================================
# Data types dictionary to enforce float32 for continuous features
# and int8 for the binary target class, keeping memory usage minimal.
DTYPE_MAPPING = {col: "float32" for col in FEATURE_COLS}
DTYPE_MAPPING[TARGET_COL] = "int8"
