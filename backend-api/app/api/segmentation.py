import io
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from app.utils.image import load_image_bytes, preprocess_for_segmentation, encode_image_to_base64
from app.services.segmentation_loader import load_brain_mri_segmentation_model

segmentation_router = APIRouter(prefix="/segmentation", tags=["segmentation"])

SEG_SIZE = (256, 256)          # the U-Net's trained input size
THRESHOLD = 0.5               # probability above which a pixel is called "tumor"


@segmentation_router.post("/brain-mri")
async def brain_mri_segmentation(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    # Load the trained U-Net. If it's missing (or TF isn't installed), fail
    # cleanly with 503 rather than returning a fake result.
    model = load_brain_mri_segmentation_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Brain-MRI segmentation model is unavailable. "
                   "Place 'brain_mri_unet.h5' in the models/ directory.",
        )

    contents = await file.read()
    image = load_image_bytes(contents)                       # RGB PIL image
    image_array = preprocess_for_segmentation(image, SEG_SIZE)  # (1,256,256,3) in [0,1]

    # Predict the per-pixel tumor probability, then threshold to a binary mask.
    prob_mask = model.predict(image_array)[0, :, :, 0]        # (256,256) floats 0..1
    binary_mask = (prob_mask > THRESHOLD)                     # (256,256) bool

    # How much of the image the model marked as tumor (a simple, honest signal).
    tumor_ratio = float(binary_mask.mean())
    tumor_detected = bool(binary_mask.any())

    # 1) Raw mask as a white-on-black PNG.
    mask_uint8 = (binary_mask.astype(np.uint8) * 255)
    mask_png = _to_png_bytes(Image.fromarray(mask_uint8, mode="L"))

    # 2) Overlay: paint the tumor region red on top of the original scan, so the
    #    result is directly viewable (like the Grad-CAM overlay for classification).
    base = np.array(image.resize(SEG_SIZE).convert("RGB"), dtype=np.uint8)
    overlay = base.copy()
    overlay[binary_mask] = [255, 0, 0]                        # red where tumor
    blended = (0.6 * base + 0.4 * overlay).astype(np.uint8)   # translucent red
    overlay_png = _to_png_bytes(Image.fromarray(blended, mode="RGB"))

    return JSONResponse({
        "modality": "brain-mri",
        "task": "segmentation",
        "tumor_detected": tumor_detected,
        "tumor_area_ratio": round(tumor_ratio, 4),
        "mask_image": encode_image_to_base64(mask_png),
        "overlay_image": encode_image_to_base64(overlay_png),
        "metadata": {
            "model_version": "unet-v1.0",
            "notes": "Brain-MRI tumor segmentation (U-Net). Pixels above 0.5 marked as tumor.",
        },
    })


def _to_png_bytes(pil_image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()
