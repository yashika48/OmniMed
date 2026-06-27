# OmniMed Backend Phases

This project is designed as a production-style, modular multi-modal medical imaging AI platform.
We break development into small, incremental phases so each milestone is clear and each component is testable.

## Phase 1: Modular FastAPI foundation

- Create a clean FastAPI application entrypoint under `backend-api/app/main.py`
- Add modular API router structure in `backend-api/app/api/`
- Add environment configuration in `backend-api/app/core/config.py`
- Add type-safe request/response schemas in `backend-api/app/schemas.py`
- Establish a model loader and handler registry for each modality
- Support Brain MRI, Chest X-ray, and Skin Lesion classification handlers
- Provide a single `/health` endpoint for deployment readiness

## Phase 2: Prediction + Grad-CAM

- Add generic image preprocessing utilities
- Implement Grad-CAM heatmap generation for all modalities
- Create handler-specific prediction logic for each modality
- Expose `/predict/{modality}` endpoint
- Return structured prediction results, probability breakdown, and Grad-CAM tile data

## Phase 3: Brain MRI segmentation

- Implement a U-Net style segmentation architecture in `backend-api/app/utils/unet.py`
- Add a Brain MRI segmentation service and endpoint under `/segmentation/brain-mri`
- Support model-based mask generation if weights are available
- Provide a fallback synthetic mask when model weights are missing for local development

## Phase 4: LLM report generation

- Add LLM report service in `backend-api/app/services/report_service.py`
- Generate medical-style reports from prediction results and clinical metadata
- Support OpenAI with environment-driven API key
- Provide fallback template-based reports when no API key is configured

## Phase 5: Production polish

- Add `requirements.txt` and documentation
- Add Docker support and environment variable configuration
- Add unit tests for API endpoints and utility functions
- Add front-end integration and demonstration dashboards

## Phase 6: Resume-ready enhancements

- Add image modality metadata and deployment documentation
- Add real segmentation visual outputs and explainability examples
- Add performance metrics and model evaluation notes
- Add multi-user authentication and patient record persistence
- Add explicit architecture diagrams and full-stack demo stories
