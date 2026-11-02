"""Script to generate evaluation plots, figures, and report tables."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    print("[FraudSense] Generating report figures and assets...")


if __name__ == "__main__":
    main()
