"""Script to build data/sample/demo_transactions.csv with 3,000 rows.

Contains:
- All 492 frauds + 2,508 random normal transactions (seed 42)
- Same 31 columns: Time, V1..V28, Amount, Class
- If data/raw/creditcard.csv exists, extracts directly from it.
- If data/raw/creditcard.csv is missing, generates a high-fidelity synthetic demo sample.

Notice: This dataset is a demo sample where frauds are intentionally over-represented
for demonstration purposes; it is NOT an evaluation set.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    RAW_DATA_FILE,
    SAMPLE_DATA_FILE,
    ALL_COLS,
    RANDOM_SEED,
)
from src.data.loader import load_raw


DEMO_DISCLAIMER = "demo sample: frauds over-represented for demonstration, not an evaluation set"


def build_demo_sample(
    output_path: Path = SAMPLE_DATA_FILE,
    target_frauds: int = 492,
    target_normals: int = 2508,
    seed: int = RANDOM_SEED,
) -> Path:
    """Extract or generate 3,000 sample transactions (492 frauds + 2,508 normals)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if RAW_DATA_FILE.exists():
        print(f"[INFO] Extracting sample from raw dataset: {RAW_DATA_FILE}...")
        df_raw = load_raw(RAW_DATA_FILE)

        fraud_df = df_raw[df_raw["Class"] == 1]
        normal_df = df_raw[df_raw["Class"] == 0]

        # Sample normals
        n_normals_to_sample = min(len(normal_df), target_normals)
        sampled_normals = normal_df.sample(n=n_normals_to_sample, random_state=seed)

        # Combine all frauds + sampled normals
        demo_df = pd.concat([fraud_df, sampled_normals], ignore_index=True)
    else:
        print("[INFO] Raw creditcard.csv not found. Generating 3,000 synthetic demo transactions...")
        rng = np.random.default_rng(seed)

        # 1. 2,508 Normal transactions
        normal_dict = {f"V{i}": rng.normal(0.0, 1.0, target_normals).astype(np.float32) for i in range(1, 29)}
        normal_dict["Time"] = rng.uniform(0.0, 172800.0, target_normals).astype(np.float32)
        normal_dict["Amount"] = rng.exponential(65.0, target_normals).astype(np.float32)
        normal_dict["Class"] = np.zeros(target_normals, dtype=np.int8)
        demo_normal = pd.DataFrame(normal_dict)

        # 2. 492 Fraudulent transactions (characteristic shifts on V10, V12, V14, V17)
        fraud_dict = {f"V{i}": rng.normal(0.0, 1.2, target_frauds).astype(np.float32) for i in range(1, 29)}
        fraud_dict["V10"] = rng.normal(-4.8, 1.6, target_frauds).astype(np.float32)
        fraud_dict["V12"] = rng.normal(-5.2, 1.6, target_frauds).astype(np.float32)
        fraud_dict["V14"] = rng.normal(-6.5, 1.8, target_frauds).astype(np.float32)
        fraud_dict["V17"] = rng.normal(-5.8, 1.7, target_frauds).astype(np.float32)
        fraud_dict["Time"] = rng.uniform(0.0, 172800.0, target_frauds).astype(np.float32)
        fraud_dict["Amount"] = rng.exponential(135.0, target_frauds).astype(np.float32)
        fraud_dict["Class"] = np.ones(target_frauds, dtype=np.int8)
        demo_fraud = pd.DataFrame(fraud_dict)

        demo_df = pd.concat([demo_normal, demo_fraud], ignore_index=True)

    # Ensure 31 columns in fixed order and shuffle rows
    demo_df = demo_df[ALL_COLS].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    demo_df.to_csv(output_path, index=False)

    frauds_count = int((demo_df["Class"] == 1).sum())
    normals_count = int((demo_df["Class"] == 0).sum())

    print(f"[OK] Built demo sample at: {output_path.resolve()}")
    print(f"     Total Rows : {len(demo_df):,d} ({frauds_count} frauds + {normals_count} normal transactions)")
    print(f"     Disclaimer : {DEMO_DISCLAIMER}")

    return output_path


if __name__ == "__main__":
    build_demo_sample()
