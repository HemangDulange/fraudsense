"""Dataset audit script for FraudSense.

This script executes from the repository root:
1. Loads the raw dataset (or demo sample if raw is missing).
2. Computes the SHA-256 cryptographic hash of the input file.
3. Audits shape, missing values, duplicates, class distribution, Amount and Time.
4. Saves structured results to:
   - artifacts/data_audit.json
   - artifacts/data_audit.md
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    RAW_DATA_FILE,
    SAMPLE_DATA_FILE,
    ARTIFACTS_DIR,
)
from src.data.loader import load_raw
from src.data.audit import (
    compute_file_sha256,
    audit_dataframe,
    save_audit_report,
)


def run_audit(file_path: Path = None):
    """Run data audit and write reports to artifacts/ directory."""
    target_file = file_path if file_path is not None else RAW_DATA_FILE

    print("\n========================================================")
    print("  [FraudSense] Dataset Quality & Imbalance Audit")
    print("========================================================")

    # If raw data file is not found, offer fallback to demo transactions
    if not target_file.exists():
        print(f"[WARN] Raw dataset not found at: {target_file}")
        if SAMPLE_DATA_FILE.exists():
            print(f"[INFO] Falling back to demo transactions: {SAMPLE_DATA_FILE}")
            target_file = SAMPLE_DATA_FILE
        else:
            # This will raise the clear FileNotFoundError explaining where to download creditcard.csv
            load_raw(target_file)

    print(f"[INFO] Auditing file: {target_file.resolve()}")

    # 1. Compute SHA-256 checksum
    print("[INFO] Computing SHA-256 checksum (streaming in chunks)...")
    file_hash = compute_file_sha256(target_file)
    print(f"[INFO] SHA-256: {file_hash}")

    # 2. Load dataset
    print("[INFO] Loading dataset (using float32 to optimize memory)...")
    df = load_raw(target_file)
    print(f"[OK] Loaded {len(df):,d} rows and {len(df.columns)} columns.")

    # 3. Perform audit
    print("[INFO] Performing audit checks...")
    audit_results = audit_dataframe(df, file_hash=file_hash)

    # 4. Save reports
    json_path = ARTIFACTS_DIR / "data_audit.json"
    md_path = ARTIFACTS_DIR / "data_audit.md"

    print(f"[INFO] Saving JSON report to: {json_path}")
    print(f"[INFO] Saving Markdown report to: {md_path}")
    save_audit_report(audit_results, json_path=json_path, md_path=md_path)

    # 5. Print headline summary
    cls_dist = audit_results["class_distribution"]
    qual = audit_results["data_quality"]
    print("\n---------------- Headline Results ----------------")
    print(f"Total Rows      : {audit_results['shape']['num_rows']:,d}")
    print(f"Normal (Class 0): {cls_dist['normal_transactions']:,d}")
    print(f"Fraud (Class 1) : {cls_dist['fraud_transactions']:,d}")
    print(f"Fraud Rate      : {cls_dist['fraud_rate_pct']:.4f}%")
    print(f"Missing Values  : {qual['total_missing_values']}")
    print(f"Duplicates      : {qual['duplicate_rows']:,d}")
    print("--------------------------------------------------\n")
    print("[OK] Audit completed successfully!")


if __name__ == "__main__":
    run_audit()
