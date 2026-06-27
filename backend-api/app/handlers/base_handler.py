from abc import ABC
from fastapi import UploadFile, HTTPException
from app.constants import MODALITY_CONFIG
from app.services.model_loader import load_model_for_modality
from app.utils.image import load_image_bytes, preprocess_image, encode_image_to_base64
from app.utils.gradcam import generate_gradcam
from app.schemas import PredictionResponse


class BaseHandler(ABC):
    """Shared prediction pipeline for every imaging modality.

    Concrete handlers declare only their `modality` (and a human-readable
    `display_name`). Labels and input size are pulled from MODALITY_CONFIG, and
    the full predict() pipeline lives here so it is defined once instead of being
    copy-pasted per modality. Adding a new modality is now: add a config entry,
    add a 3-line handler, register it in the factory.
    """

    modality: str
    display_name: str = ""
    labels: list[str]
    input_size: tuple[int, int]

    def __init__(self) -> None:
        config = MODALITY_CONFIG[self.modality]
        self.labels = config["labels"]
        self.input_size = config["input_size"]

    async def predict(self, file: UploadFile) -> PredictionResponse:
        contents = await file.read()
        image = load_image_bytes(contents)
        image_array = preprocess_image(image, self.input_size)

        model = load_model_for_modality(self.modality)
        if model is None:
            raise HTTPException(status_code=503, detail=f"Model for {self.modality} is unavailable.")

        raw_predictions = model.predict(image_array)[0]
        predicted_index = int(raw_predictions.argmax())
        predicted_label = self.labels[predicted_index]
        probabilities = {self.labels[i]: float(raw_predictions[i]) for i in range(len(self.labels))}

        grad_cam = generate_gradcam(model, image_array, predicted_index)
        grad_cam_base64 = encode_image_to_base64(grad_cam)

        return PredictionResponse(
            modality=self.modality,
            prediction=predicted_label,
            confidence=float(raw_predictions[predicted_index]),
            probabilities=probabilities,
            grad_cam_image=grad_cam_base64,
            metadata={
                "model_version": "v1.0",
                "notes": f"{self.display_name or self.modality} modality with Grad-CAM.",
            },
        )
