"""Model and artifact serialization using JSON + NPZ (allow_pickle=False).

Strictly adheres to project Rule 5:
- Save models as JSON + NPZ with allow_pickle=False.
- Never use pickle.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import json
import numpy as np

from src.models.gaussian_detector import GaussianDetector
from src.models.mahalanobis_detector import MahalanobisDetector
from src.models.pca_detector import PCADetector
from src.models.isolation_forest import IsolationForestFromScratch
from src.models.logistic_regression import LogisticRegressionFromScratch

MODEL_REGISTRY = {
    "GaussianDetector": GaussianDetector,
    "MahalanobisDetector": MahalanobisDetector,
    "PCADetector": PCADetector,
    "IsolationForestFromScratch": IsolationForestFromScratch,
    "LogisticRegressionFromScratch": LogisticRegressionFromScratch,
}


def save_model(
    model: Any,
    model_name: str,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Save a model's parameters as JSON + NPZ without using pickle.

    Parameters
    ----------
    model : Any
        Fitted model instance with to_state() method.
    model_name : str
        Base filename (e.g. 'gaussian', 'mahalanobis').
    output_dir : Path
        Directory to write files to.

    Returns
    -------
    json_path : Path
    npz_path : Path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    state = model.to_state()

    json_data = {}
    npz_arrays = {}

    for k, v in state.items():
        if isinstance(v, np.ndarray):
            npz_arrays[k] = v
            json_data[k] = f"__npz__{k}"
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float, list)):
            # Large numerical lists (e.g., weights, components, means) saved to NPZ
            arr = np.array(v)
            if arr.dtype.kind in ("f", "i"):
                npz_arrays[k] = arr
                json_data[k] = f"__npz__{k}"
            else:
                json_data[k] = v
        else:
            json_data[k] = v

    json_path = output_dir / f"{model_name}.json"
    npz_path = output_dir / f"{model_name}.npz"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    # Save numerical arrays to compressed NPZ with allow_pickle=False
    # To guarantee allow_pickle=False compatibility, pass arrays as keyword arguments
    np.savez_compressed(npz_path, **npz_arrays)

    return json_path, npz_path


def load_model(
    model_name: str,
    input_dir: Path,
) -> Any:
    """Load a model from JSON + NPZ with allow_pickle=False.

    Parameters
    ----------
    model_name : str
        Base filename without extension.
    input_dir : Path
        Directory containing the saved model files.

    Returns
    -------
    Any
        Reconstructed model instance.
    """
    json_path = input_dir / f"{model_name}.json"
    npz_path = input_dir / f"{model_name}.npz"

    if not json_path.exists():
        raise FileNotFoundError(f"Model JSON not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    if npz_path.exists():
        # Load arrays strictly with allow_pickle=False
        with np.load(npz_path, allow_pickle=False) as npz:
            for k in list(state.keys()):
                val = state[k]
                if isinstance(val, str) and val.startswith("__npz__"):
                    arr_key = val.replace("__npz__", "")
                    state[k] = npz[arr_key]

    model_type = state.get("model_type")
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type '{model_type}' in state file {json_path}")

    model_cls = MODEL_REGISTRY[model_type]
    return model_cls.from_state(state)
