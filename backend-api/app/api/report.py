from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.services.report_service import generate_llm_report

report_router = APIRouter(prefix="/report", tags=["report"])


class ReportRequest(BaseModel):
    modality: str
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    metadata: dict[str, str] = Field(default_factory=dict)
    notes: str | None = None


class ReportResponse(BaseModel):
    modality: str
    report: str


@report_router.post("/generate", response_model=ReportResponse)
def generate_report(request: ReportRequest) -> ReportResponse:
    if request.confidence < 0 or request.confidence > 1:
        raise HTTPException(status_code=400, detail="Confidence score must be between 0 and 1.")

    report_text = generate_llm_report(
        request.modality,
        request.prediction,
        request.confidence,
        request.probabilities,
        request.metadata,
        extra_notes=request.notes,
    )

    return ReportResponse(modality=request.modality, report=report_text)
