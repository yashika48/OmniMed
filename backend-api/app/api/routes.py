from fastapi import APIRouter
from app.api.predict import predict_router
from app.api.segmentation import segmentation_router
from app.api.report import report_router

api_router = APIRouter()
api_router.include_router(predict_router)
api_router.include_router(segmentation_router)
api_router.include_router(report_router)

@api_router.get("/health", tags=["health"])
def health_check():
    return {"status": "ok", "detail": "OmniMed backend is available."}
