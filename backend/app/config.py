"""Application configuration loaded from environment variables.

All settings have sensible development defaults so the application boots with no
configuration.  Override anything via a ``.env`` file (see ``.env.example``) or
real environment variables for production deployments.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- General ---------------------------------------------------------
    app_name: str = "Smart Meeting"
    environment: str = Field(default="development")
    debug: bool = Field(default=True)

    # --- Security / auth -------------------------------------------------
    # IMPORTANT: override JWT_SECRET_KEY in production with a long random value.
    jwt_secret_key: str = Field(
        default="dev-insecure-secret-change-me-in-production-please"
    )
    jwt_algorithm: str = "HS256"
    # 7 day expiry as specified in the requirements.
    access_token_expire_minutes: int = 60 * 24 * 7
    bcrypt_rounds: int = 12

    # --- Database --------------------------------------------------------
    # Defaults to a local SQLite file. Point at PostgreSQL in production, e.g.
    # postgresql+psycopg://user:pass@host:5432/smartmeeting
    database_url: str = "sqlite:///./smart_meeting.db"

    # --- CORS ------------------------------------------------------------
    cors_origins: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:4173",
        ]
    )

    # --- Storage ---------------------------------------------------------
    audio_storage_dir: str = "./data/audio"

    # --- Audio / transcription ------------------------------------------
    audio_sample_rate: int = 16000
    audio_channels: int = 1

    # Whisper model sizes for the two-pass pipeline.
    # Defaults favour machines without a GPU: int8 keeps memory under control so
    # the API process is not OOM-killed while loading the finalization model.
    whisper_live_model: str = "base"
    whisper_final_model: str = "medium"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    # Dialect / language handling. "hil" == Hiligaynon.
    whisper_default_language: str = "hil"

    # --- LLM (BART / mBART) ---------------------------------------------
    bart_model: str = "facebook/bart-large-cnn"
    mbart_model: str = "facebook/mbart-large-50-many-to-many-mmt"
    # When true, allow lightweight non-ML fallbacks (extractive summary) so the
    # feature works even without the heavy model weights downloaded.
    allow_llm_fallback: bool = True

    # --- Rate limiting ---------------------------------------------------
    rate_limit_auth: str = "20/minute"
    rate_limit_ai: str = "30/minute"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, v):
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
