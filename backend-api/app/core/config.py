from pathlib import Path
import os

# Try to use pydantic BaseSettings if available; otherwise provide a lightweight fallback.
try:
    from pydantic import BaseSettings

    class Settings(BaseSettings):
        app_name: str = "OmniMed"
        environment: str = "development"
        model_dir: Path = Path("models")
        allowed_origins: list[str] = ["*"]
        openai_api_key: str | None = None
        openai_model: str = "gpt-4o-mini"
        max_upload_size_mb: int = 20

        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"

    settings = Settings()
except Exception:
    class Settings:
        def __init__(self):
            self.app_name = os.getenv("APP_NAME", "OmniMed")
            self.environment = os.getenv("ENVIRONMENT", "development")
            self.model_dir = Path(os.getenv("MODEL_DIR", "models"))
            allowed = os.getenv("ALLOWED_ORIGINS", "*")
            self.allowed_origins = allowed.split(",") if isinstance(allowed, str) else ["*"]
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            try:
                self.max_upload_size_mb = int(os.getenv("MAX_UPLOAD_SIZE_MB", "20"))
            except ValueError:
                self.max_upload_size_mb = 20

    settings = Settings()
