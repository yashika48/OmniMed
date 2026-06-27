from pydantic import BaseModel, Field
from typing import Dict, Optional


class PredictionResponse(BaseModel):
    modality: str = Field(..., description="Imaging modality used for prediction")
    prediction: str = Field(..., description="Top predicted class label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    probabilities: Dict[str, float] = Field(..., description="Probability breakdown across all classes")
    grad_cam_image: Optional[str] = Field(None, description="Base64-encoded Grad-CAM heatmap image")
    metadata: Dict[str, str] = Field(default_factory=dict, description="Optional modality metadata")


class HealthResponse(BaseModel):
    status: str
    detail: str
