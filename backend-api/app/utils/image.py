import io
import base64
from typing import Tuple
from PIL import Image
import numpy as np
from fastapi import HTTPException
from app.core.config import settings


def load_image_bytes(image_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert("RGB")
        return image
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to read image: {exc}")


def preprocess_image(image: Image.Image, target_size: Tuple[int, int]) -> np.ndarray:
    image = image.resize(target_size)
    image_array = np.array(image, dtype=np.float32)
    # The saved model already contains its own rescaling/normalization layers.
    # Passing raw pixel values preserves the model's expected input scale.
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def encode_image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def preprocess_for_segmentation(image: Image.Image, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """Preprocess an image for the U-Net segmentation model.

    Unlike the EfficientNet classifiers (which rescale internally, so they take
    raw 0-255 pixels), the U-Net was trained on pixels scaled to [0, 1] at
    256x256. So segmentation must resize to 256x256 AND divide by 255 -- using
    the classifier preprocessing here would give a wrong-sized, wrong-scaled
    input and garbage masks.
    """
    image = image.resize(target_size)
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array
