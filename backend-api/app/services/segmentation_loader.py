from pathlib import Path
from app.constants import MODEL_DIR

# Cache the loaded model so we only read it from disk once (lazy loading),
# matching how the classification models are handled.
_segmentation_model_cache = None


def load_brain_mri_segmentation_model():
    """Load the trained brain-MRI U-Net, or return None if unavailable.

    The model is loaded with compile=False: the U-Net was trained with custom
    Dice loss / Dice / IoU objects, which Keras would otherwise need to be told
    about on load. For inference we don't need the loss or metrics at all, so
    compile=False skips them entirely and loads just the architecture + weights.
    """
    global _segmentation_model_cache
    if _segmentation_model_cache is not None:
        return _segmentation_model_cache

    model_path = MODEL_DIR / "brain_mri_unet.h5"
    if not model_path.exists():
        return None

    try:
        import tensorflow as tf
    except ModuleNotFoundError:
        return None

    _segmentation_model_cache = tf.keras.models.load_model(model_path, compile=False)
    return _segmentation_model_cache
