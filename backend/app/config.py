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
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ]
    )

    # --- Storage ---------------------------------------------------------
    audio_storage_dir: str = "./data/audio"
    # Redis holds a *rolling* live PCM window + cached WAV + session meta.
    # Full multi-hour PCM is appended on disk (see audio.append_raw_pcm).
    redis_url: str = "redis://127.0.0.1:6379/0"
    # TTL for meeting audio keys in Redis (seconds). 0 = no expiry.
    # Default 24h — long enough for reconnect, short enough to avoid landfill.
    redis_audio_ttl_seconds: int = 60 * 60 * 24
    # How much recent PCM Redis keeps for resume (seconds). ~3 min ≈ 5.8 MB.
    redis_live_buffer_seconds: float = 180.0
    # Soft cap for a single live recording (hours). Used for warnings / docs;
    # disk continues buffering until this limit (0 = unlimited).
    max_meeting_hours: float = 16.0
    # Abandoned live sessions (no finalize) are cleaned after this many seconds.
    abandoned_session_seconds: int = 60 * 5
    # How often the background janitor scans for abandoned sessions.
    session_janitor_interval_seconds: float = 60.0
    # Drop old live windows when ASR backlog exceeds this many hops.
    # Higher default reduces live word-loss when Whisper is slower than realtime.
    live_asr_max_backlog_windows: int = 12
    # When over backlog, skip at most this many hops per wake (partial catch-up
    # instead of jumping to the live edge and losing mid-meeting captions).
    live_asr_backpressure_skip_hops: int = 2

    # --- Audio / transcription ------------------------------------------
    audio_sample_rate: int = 16000
    audio_channels: int = 1

    # Whisper model sizes for the two-pass pipeline.
    # Live + final default to faster-whisper (reliable on CPU). Optional HF
    # fine-tune can be enabled with WHISPER_FINAL_BACKEND=huggingface.
    whisper_live_model: str = "small"
    whisper_final_model: str = "medium"
    # Hugging Face id (or local path) for an optional fine-tuned final ASR model.
    whisper_hiligaynon_model: str = "rbcurzon/whisper-medium-ph"
    # "faster-whisper" (default, stable) or "huggingface" (fine-tuned checkpoint).
    whisper_final_backend: str = "faster-whisper"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    # Meeting language label (Hiligaynon). Decode language for Whisper is ``tl``.
    whisper_default_language: str = "hil"
    # Live Whisper decode language (``tl`` = Tagalog/PH). Live stays forced for
    # low-latency captions; final pass uses ``whisper_final_language_mode``.
    whisper_decode_language: str = "tl"
    # Final-pass language: "auto" (detect), "forced" (whisper_decode_language),
    # or "prefer_forced" (forced first, auto retry if coverage is poor).
    whisper_final_language_mode: str = "auto"
    # Final-pass VAD: aggressive VAD was dropping long spans of real speech
    # (especially clipped mic audio / mixed EN+PH). Default off for coverage.
    whisper_final_vad_filter: bool = False
    # Live caption windowing: 10s buffer overlapping by 5s (hop = 5s).
    whisper_live_window_seconds: float = 10.0
    whisper_live_hop_seconds: float = 5.0
    # Emit a short first caption quickly so recording does not feel stuck
    # waiting for the full 10s window (steady-state still uses 10s/5s).
    whisper_live_warmup_seconds: float = 2.5
    # Final ASR chunk size for multi-hour recordings (seconds of audio per pass).
    whisper_final_chunk_seconds: float = 600.0
    whisper_final_chunk_overlap_seconds: float = 15.0
    # If final segments cover less than this fraction of audio duration, retry
    # with alternate language / no-VAD settings before accepting the result.
    whisper_final_min_coverage: float = 0.55
    # How often to persist live segments to SQLite during long meetings.
    # 1 = every window; 12 ≈ once per minute with a 5s hop.
    live_segment_persist_every: int = 12
    # WebSocket keepalive interval so proxies don't drop 8h+ sessions.
    ws_keepalive_seconds: float = 25.0
    # Max Whisper backends retained in memory (LRU). Live + final usually need 2.
    whisper_model_cache_size: int = 2
    # Prefer accumulated live captions over the final ASR pass when the final
    # word count falls below ``live_words * ratio`` (and live has enough words).
    # Example: 0.6 means "final has < 60% of live words" → keep live text.
    live_caption_prefer_ratio: float = 0.6
    # Minimum live-caption word count before the preference rule can fire
    # (avoids overriding a short/empty final with a tiny live hallucination).
    live_caption_prefer_min_words: int = 12
    # ffmpeg subprocess timeout when decoding non-WAV uploads (seconds).
    ffmpeg_timeout_seconds: float = 120.0

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
