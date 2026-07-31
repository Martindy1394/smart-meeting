"""Application configuration loaded from environment variables.

All settings have sensible development defaults so the application boots with no
configuration.  Override anything via a ``.env`` file (see ``.env.example``) or
real environment variables for production deployments.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Annotated, List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


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
    # Short-lived access token (minutes). Use refresh tokens for session length.
    access_token_expire_minutes: int = 30
    # Refresh token lifetime (days). Revocable via jti denylist / logout.
    refresh_token_expire_days: int = 7
    bcrypt_rounds: int = 12
    # Fernet url-safe key for audio encryption-at-rest (optional).
    # python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    data_encryption_key: str = ""

    # --- Database --------------------------------------------------------
    # Defaults to a local SQLite file. Point at PostgreSQL in production, e.g.
    # postgresql+psycopg://user:pass@host:5432/smartmeeting
    database_url: str = "sqlite:///./smart_meeting.db"

    # --- CORS ------------------------------------------------------------
    # NoDecode: allow comma-separated CORS_ORIGINS in .env (not only JSON arrays).
    cors_origins: Annotated[List[str], NoDecode] = Field(
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
    # Mark meetings stuck in ``processing`` as failed after this many seconds
    # (Whisper on CPU can take a long time; keep this above worst-case ASR).
    processing_stale_seconds: int = 60 * 45
    # TTL for the single-writer live WebSocket lock in Redis.
    live_session_lock_ttl_seconds: int = 90
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
    # Live uses faster-whisper. Final prefers a fine-tuned Hiligaynon/PH
    # checkpoint when backend is ``auto`` / ``huggingface``.
    whisper_live_model: str = "small"
    whisper_final_model: str = "medium"
    # Your own Hiligaynon fine-tune (HF repo id or local transformers folder).
    # Tried first for hil meetings when set; leave empty until you have one.
    # Recommended source: UP-DSP PLD Hiligaynon (~41h) via
    # scripts/hiligaynon_asr/import_pld.py — see docs/PLD.md.
    whisper_hiligaynon_fine_tuned_model: str = ""
    # Philippine / Hiligaynon-capable HF fallback (used when no custom fine-tune,
    # or after the custom checkpoint fails).
    whisper_hiligaynon_model: str = "rbcurzon/whisper-medium-ph"
    # Optional faster-whisper / CTranslate2 export of a Hiligaynon fine-tune
    # for live captions (local dir or compatible HF repo). Empty = use live model.
    whisper_live_hiligaynon_model: str = ""
    # Your own Tagalog fine-tune (HF/local). Tried first for tl/fil meetings.
    whisper_tagalog_fine_tuned_model: str = ""
    # Tagalog-focused HF model (FLEURS fil_ph fine-tune). Used after custom
    # Tagalog fine-tune and before the broader Philippine medium fallback.
    whisper_tagalog_model: str = "LWobole/whisper-small-tagalog"
    # Optional CT2 Tagalog fine-tune for live captions.
    whisper_live_tagalog_model: str = ""
    # auto | huggingface | faster-whisper
    # auto: for Hiligaynon/Tagalog/PH meetings try HF candidates, then FW.
    whisper_final_backend: str = "auto"
    # Live captions backend: whisper | rnnt | auto
    # auto = NeMo FastConformer-RNNT for PH/Hiligaynon-biased meetings when
    # installed, else faster-whisper. Final pass always stays on Whisper.
    whisper_live_backend: str = "auto"
    # NeMo .nemo path or HF repo for live RNN-T (Tagalog FastConformer-Hybrid).
    rnnt_live_model: str = "NCSpeech/stt_tl_fastconformer_hybrid_large"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    # Meeting language label is always ``auto`` in the product UI. This setting
    # is the ASR bias used when resolving ``auto`` (Hiligaynon prompts + PH
    # models). Hiligaynon uses Whisper auto-detect — never forced Tagalog.
    # Meeting ``auto`` still resolves to this default at transcription time.
    whisper_default_language: str = "hil"
    # Optional prompts — keep short to avoid Whisper echoing them.
    # Used for auto-detect and mixed PH board meetings (Iloilo / Hiligaynon-heavy).
    whisper_initial_prompt: str = (
        "Board meeting discussion in Hiligaynon and English."
    )
    whisper_tagalog_initial_prompt: str = (
        "Board meeting discussion in Tagalog and English."
    )
    # Keep SHORT — long word lists get echoed into the transcript by Whisper.
    # A few stable Ilonggo markers bias decode without a word-list dump.
    whisper_hiligaynon_initial_prompt: str = (
        "Diskusyon sa board meeting sa Hiligaynon kag English. "
        "Wala gid, indi, sang, kag, amo."
    )
    # Final-pass language for non-Tagalog / non-Hiligaynon fallbacks.
    # ``prefer_forced`` = forced language first, auto retry if coverage is poor.
    whisper_final_language_mode: str = "auto"
    # Only used when meeting language is explicitly Tagalog (API/legacy).
    whisper_tagalog_final_language_mode: str = "prefer_forced"
    # Hiligaynon: Whisper auto-detect (no hil token; never force Tagalog ``tl``).
    whisper_hiligaynon_final_language_mode: str = "auto"
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
    whisper_model_cache_size: int = 3
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
    # Optional LoRA-merged mBART checkpoint fine-tuned for Tagalog/Hiligaynon→EN.
    # Train on native ``tl_XX`` (see docs/FINE_TUNE_MBART_PH.md). Hiligaynon has
    # no mBART token — fine-tune can only proxy it under ``tl_XX``.
    mbart_ph_finetuned_model: str = ""
    # auto | nllb | mbart — Tagalog/mixed backend preference.
    # auto: fine-tuned mBART first when configured, otherwise NLLB.
    # Stock mBART Tagalog uses native ``tl_XX`` (not id_ID) — see docs/MBART_PH_AUDIT.md.
    # Hiligaynon lines are routed to Google Translate first, not mBART.
    ph_translate_backend: str = "auto"
    # NLLB has a real Tagalog code (tgl_Latn); used for Tagalog → English.
    nllb_model: str = "facebook/nllb-200-distilled-600M"
    # Hiligaynon → English via Google Cloud Translation (``hil`` code, mid-2024).
    # NLLB/mBART do not cover Hiligaynon; ceb_Latn is only an emergency fallback.
    google_translate_enabled: bool = True
    google_cloud_project: str = ""
    # google | nllb — when Google is unavailable, fall back to NLLB ceb_Latn.
    hil_translate_fallback: str = "nllb"
    # When true, allow lightweight non-ML fallbacks (extractive summary) so the
    # feature works even without the heavy model weights downloaded.
    allow_llm_fallback: bool = True
    # Divide-and-conquer summarization: keep each topic chunk under this many
    # BART input tokens (model limit is 1024; leave headroom for special tokens).
    bart_max_input_tokens: int = 960
    # Start a new topic when consecutive discourse-unit TF cosine similarity
    # drops below this threshold (0–1). Lower = fewer / larger topics.
    bart_topic_similarity_threshold: float = 0.22
    # Minimum discourse units in a topic before a similarity drop can split.
    bart_topic_min_units: int = 2

    # --- VAD / confidence filtering --------------------------------------
    vad_enabled: bool = True
    # auto | webrtcvad | energy
    vad_backend: str = "auto"
    vad_aggressiveness: int = 2  # webrtcvad 0–3
    vad_min_speech_ratio: float = 0.15
    vad_live_min_rms: float = 0.006
    vad_final_min_rms: float = 0.004
    # Drop / flag Whisper segments below these thresholds.
    asr_min_avg_logprob: float = -1.0
    asr_max_no_speech_prob: float = 0.6
    asr_flag_avg_logprob: float = -0.6
    asr_flag_no_speech_prob: float = 0.45
    # Session language lock: detect on first N seconds, then reuse.
    asr_language_lock_seconds: float = 8.0
    asr_language_relock_no_speech_prob: float = 0.85
    asr_language_min_confidence: float = 0.45

    # --- Background jobs / retention -------------------------------------
    jobs_use_redis_queue: bool = True
    jobs_inline_worker: bool = True
    # Purge finalized WAV/PCM after N days (0 = keep forever). Transcripts stay.
    audio_retention_days: int = 30

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
