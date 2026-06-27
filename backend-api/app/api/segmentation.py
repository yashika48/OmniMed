import io
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from app.utils.image import load_image_bytes, preprocess_image, encode_image_to_base64
from app.utils.unet import build_unet
from app.services.segmentation_loader import load_brain_mri_segmentation_model
from app.constants import MODEL_DIR

segmentation_router = APIRouter(prefix="/segmentation", tags=["segmentation"])


@segmentation_router.post("/brain-mri")
async def brain_mri_segmentation(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    contents = await file.read()
    image = load_image_bytes(contents)
    image_array = preprocess_image(image, (224, 224))

    model = load_brain_mri_segmentation_model()
    if model is None:
        model = build_unet((224, 224, 3))

    prediction_mask = model.predict(image_array)[0, :, :, 0]
    mask = (prediction_mask > 0.5).astype("uint8") * 255

    mask_image = Image.fromarray(mask)
    buffer = io.BytesIO()
    mask_image.save(buffer, format="PNG")
    buffer.seek(0)

    return JSONResponse({
        "modality": "brain-mri",
        "mask_image": encode_image_to_base64(buffer.getvalue()),
    })
