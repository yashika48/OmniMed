from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import api_router
from app.core.config import settings

app = FastAPI(
    title="OmniMed Multi-Modal Medical Imaging Platform",
    version="0.1.0",
    description="Modular FastAPI backend for Brain MRI, Chest X-ray, and Skin Lesion imaging with Grad-CAM and LLM report generation.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")
