from pathlib import Path

MODEL_DIR = Path("models")

MODALITY_CONFIG = {
    "brain-mri": {
        "model_file": "brain_mri_classifier.h5",
        "labels": ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"],
        "input_size": (224, 224),
        "segmentation_file": "brain_mri_unet.h5",
    },
    "chest-xray": {
        "model_file": "chest_xray_classifier.h5",
        "labels": ["normal", "pneumonia"],
        "input_size": (224, 224),
    },
    "skin-lesion": {
        "model_file": "skin_lesion_classifier.h5",
        "labels": ["benign", "malignant"],
        "input_size": (224, 224),
    },
}

ALLOWED_MODALITIES = list(MODALITY_CONFIG.keys())
