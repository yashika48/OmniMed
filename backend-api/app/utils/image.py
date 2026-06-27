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
