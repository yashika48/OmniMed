from typing import Dict, Optional
from app.core.config import settings


def build_report_text(modality: str, prediction: str, confidence: float, probabilities: Dict[str, float], metadata: Dict[str, str], extra_notes: Optional[str] = None) -> str:
    lines = [
        f"OmniMed Clinical Report",
        f"Modality: {modality}",
        f"Prediction: {prediction}",
        f"Confidence: {confidence * 100:.1f}%",
        "",
        "Probability Breakdown:",
    ]
    for label, value in probabilities.items():
        lines.append(f" - {label}: {value * 100:.1f}%")

    if metadata:
        lines.append("")
        lines.append("Metadata:")
        for key, value in metadata.items():
            lines.append(f" - {key}: {value}")

    if extra_notes:
        lines.extend(["", "Clinician Notes:", f"{extra_notes}"])

    lines.append("")
    lines.append("Interpretation:")
    lines.append(
        "The OmniMed platform provides a preliminary diagnostic interpretation. "
        "Clinicians should cross-verify the findings with imaging reports and clinical data."
    )
    return "\n".join(lines)


def generate_llm_report(modality: str, prediction: str, confidence: float, probabilities: Dict[str, float], metadata: Dict[str, str], extra_notes: Optional[str] = None) -> str:
    if not settings.openai_api_key:
        return build_report_text(modality, prediction, confidence, probabilities, metadata, extra_notes)

    try:
        import openai
        openai.api_key = settings.openai_api_key
        prompt_lines = [
            f"Write a concise clinical summary for a {modality} imaging study.",
            f"Prediction: {prediction}",
            f"Confidence: {confidence * 100:.1f}%.",
            "Probability breakdown:",
        ]
        for label, value in probabilities.items():
            prompt_lines.append(f"{label}: {value * 100:.1f}%")
        if metadata:
            prompt_lines.append("Metadata:")
            for key, value in metadata.items():
                prompt_lines.append(f"{key}: {value}")
        if extra_notes:
            prompt_lines.append(f"Clinician Notes: {extra_notes}")

        prompt_lines.append(
            "Write the final report in professional, clinical language, suitable for inclusion in the patient's medical chart."
        )
        response = openai.ChatCompletion.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": "\n".join(prompt_lines)}],
            temperature=0.2,
            max_tokens=350,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return build_report_text(modality, prediction, confidence, probabilities, metadata, extra_notes)
