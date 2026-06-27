from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Any
from app.constants import ALLOWED_MODALITIES
from app.handlers import get_handler
from app.schemas import PredictionResponse
from app.core.config import settings

predict_router = APIRouter(prefix="/predict", tags=["predict"])


@predict_router.post("/{modality}", response_model=PredictionResponse)
async def predict_modality(modality: str, file: UploadFile = File(...)) -> Any:
    modality = modality.lower()
    if modality not in ALLOWED_MODALITIES:
        raise HTTPException(status_code=404, detail=f"Unsupported modality '{modality}'. Supported modalities: {ALLOWED_MODALITIES}")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    handler = get_handler(modality)
    prediction_result = await handler.predict(file)
    return prediction_result
