import os
from pathlib import Path
from typing import Any
from app.constants import MODEL_DIR, MODALITY_CONFIG

_model_cache: dict[str, Any] = {}


def _get_model_path(modality: str) -> Path:
    config = MODALITY_CONFIG[modality]
    primary_path = MODEL_DIR / config["model_file"]

    # Support the legacy project layout where a single trained brain MRI model
    # lives at the backend root as brain_tumor_efficientnet.h5.
    if modality == "brain-mri":
        legacy_path = Path("brain_tumor_efficientnet.h5")
        if not primary_path.exists() and legacy_path.exists():
            return legacy_path

    return primary_path


def load_model_for_modality(modality: str) -> Any | None:
    if modality in _model_cache:
        return _model_cache[modality]

    model_path = _get_model_path(modality)
    if not model_path.exists():
        return None

    try:
        import tensorflow as tf
    except ModuleNotFoundError:
        # TensorFlow not installed in this environment; return None so
        # the handlers can respond with a clear 503 and the server stays up.
        return None

    model = tf.keras.models.load_model(model_path)
    _model_cache[modality] = model
    return model


def has_model_for_modality(modality: str) -> bool:
    return _get_model_path(modality).exists()
