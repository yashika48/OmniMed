from pathlib import Path
from app.constants import MODEL_DIR


def load_brain_mri_segmentation_model() -> object | None:
    segmentation_model_path = MODEL_DIR / "brain_mri_unet.h5"
    if not segmentation_model_path.exists():
        return None

    try:
        import tensorflow as tf
    except ModuleNotFoundError:
        return None

    return tf.keras.models.load_model(segmentation_model_path)
