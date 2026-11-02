"""Pytest configuration for FraudSense.

This file automatically runs before tests are executed and adds
the repository root directory to Python's module search path (sys.path).
This ensures that imports like `import src...` work consistently
from any working directory.
"""

import sys
from pathlib import Path

# Get the absolute path to the project root directory
REPO_ROOT = Path(__file__).resolve().parent

# Add the project root to sys.path if not already present
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
