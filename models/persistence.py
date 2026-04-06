"""
Model persistence — save and load trained models using joblib.

Saved models are stored in data/saved_models/.
"""
import os
import joblib
from datetime import datetime
from utils.logger import log

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "saved_models")


def _ensure_dir():
    os.makedirs(MODELS_DIR, exist_ok=True)


def save_model(model, model_name: str, metadata: dict | None = None) -> str:
    """
    Save a trained model to disk.

    Args:
        model:       The model object (SVD, rules DataFrame, similarity matrix, etc.)
        model_name:  Short identifier, e.g. "svd", "fpgrowth_rules"
        metadata:    Optional dict of extra information to store alongside

    Returns:
        Absolute path to the saved file
    """
    _ensure_dir()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}.joblib"
    filepath = os.path.join(MODELS_DIR, filename)

    payload = {
        "model": model,
        "model_name": model_name,
        "saved_at": timestamp,
        "metadata": metadata or {},
    }

    joblib.dump(payload, filepath)
    log(f"Model saved: {filepath}")
    return filepath


def load_model(filepath: str):
    """
    Load a previously saved model.

    Args:
        filepath: Path to the .joblib file

    Returns:
        The model object (unwrapped from the payload dict)
    """
    payload = joblib.load(filepath)
    log(f"Model loaded: {filepath} (saved at {payload.get('saved_at', '?')})")
    return payload["model"]


def load_latest_model(model_name: str):
    """
    Load the most recently saved model matching the given name.

    Args:
        model_name: Short identifier, e.g. "svd"

    Returns:
        The model object, or None if not found
    """
    _ensure_dir()

    candidates = sorted([
        f for f in os.listdir(MODELS_DIR)
        if f.startswith(model_name) and f.endswith(".joblib")
    ])

    if not candidates:
        log(f"No saved model found for '{model_name}'")
        return None

    latest = os.path.join(MODELS_DIR, candidates[-1])
    return load_model(latest)


def list_saved_models() -> list[dict]:
    """List all saved models with metadata."""
    _ensure_dir()

    models = []
    for f in sorted(os.listdir(MODELS_DIR)):
        if f.endswith(".joblib"):
            path = os.path.join(MODELS_DIR, f)
            try:
                payload = joblib.load(path)
                models.append({
                    "filename": f,
                    "model_name": payload.get("model_name", "?"),
                    "saved_at": payload.get("saved_at", "?"),
                    "metadata": payload.get("metadata", {}),
                })
            except Exception:
                models.append({"filename": f, "model_name": "?", "error": "could not load"})

    return models
