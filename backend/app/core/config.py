"""
Traffic Monitor FastAPI Backend
Core configuration using Pydantic Settings.
"""

from pydantic_settings import BaseSettings
from pathlib import Path
from typing import List


class Settings(BaseSettings):
    # App
    APP_NAME: str = "Traffic Monitor API"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:5000",
    ]

    # Storage
    MODELS_DIR: Path = Path("models_storage")
    UPLOADS_DIR: Path = Path("uploads")

    # Detection defaults
    CONF_THRESHOLD: float = 0.35
    LINE_POSITION: float = 0.55
    MAX_FPS: int = 30
    TRACKER_TYPE: str = "bytetrack"  # bytetrack | botsort

    # Auto-load model
    DEFAULT_MODEL: str = "best.pt"

    # Congestion detection
    CONGESTION_VEHICLE_THRESHOLD: int = 10
    CONGESTION_STABLE_DURATION: float = 5.0

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()

settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
