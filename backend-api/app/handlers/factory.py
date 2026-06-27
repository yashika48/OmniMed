from app.handlers.brain_mri import BrainMRIHandler
from app.handlers.chest_xray import ChestXRayHandler
from app.handlers.skin_lesion import SkinLesionHandler
from app.constants import ALLOWED_MODALITIES
from fastapi import HTTPException

_HANDLER_MAP = {
    "brain-mri": BrainMRIHandler,
    "chest-xray": ChestXRayHandler,
    "skin-lesion": SkinLesionHandler,
}


def get_handler(modality: str):
    if modality not in ALLOWED_MODALITIES:
        raise HTTPException(status_code=404, detail=f"Unsupported modality '{modality}'")
    return _HANDLER_MAP[modality]()
