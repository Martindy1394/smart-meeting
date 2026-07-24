"""FastAPI application entry point for Smart Meeting."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import settings
from .database import init_db
from .routers import ai, auth, meetings
from .services import asr, audio, live_metrics, llm, redis_store
from .ws import transcription as ws_transcription

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("smart_meeting")

# /workspace/frontend/dist when running from the monorepo layout.
_FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio

    from .services import janitor, transcription as transcription_svc

    logger.info("Starting %s v%s (%s)", settings.app_name, __version__, settings.environment)
    env = settings.environment.lower()
    insecure_jwt = (
        "change-me" in settings.jwt_secret_key.lower()
        or settings.jwt_secret_key.startswith("dev-insecure")
    )
    if env not in {"development", "dev", "test"} and insecure_jwt:
        logger.error(
            "Refusing insecure JWT_SECRET_KEY defaults outside development. "
            "Set a long random JWT_SECRET_KEY before running in production."
        )
        raise RuntimeError("Insecure JWT_SECRET_KEY for non-development environment")
    init_db()
    if redis_store.is_available():
        logger.info(
            "Redis ready (%s) — rolling live buffer %.0fs, TTL %ss",
            settings.redis_url,
            settings.redis_live_buffer_seconds,
            settings.redis_audio_ttl_seconds,
        )
    else:
        logger.warning(
            "Redis not available at %s — disk PCM continues; reconnect resume limited.",
            settings.redis_url,
        )
    # Warm the live Whisper model in the background so Start recording
    # does not stall on first-caption model load.
    if transcription_svc.is_available():
        asyncio.create_task(asyncio.to_thread(transcription_svc.warm_live_model))
    stop_janitor = asyncio.Event()
    janitor_task = asyncio.create_task(janitor.janitor_loop(stop_janitor))
    yield
    stop_janitor.set()
    janitor_task.cancel()
    try:
        await janitor_task
    except (Exception, asyncio.CancelledError):
        pass
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version=__version__,
    description="Minute-making platform: Whisper ASR transcription, BART "
    "summarization, and mBART translation.",
    lifespan=lifespan,
)

# CORS — in development, accept any localhost / Cursor preview origin so
# port-forwarded browsers don't fail preflight when the tunnel host differs.
if settings.debug:
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|0\.0\.0\.0|.*\.cursor\.sh|.*\.cursorapi\.com)(:\d+)?",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/api/health", tags=["system"])
def health():
    from .services import rnnt as rnnt_svc
    from .services import transcription as transcription_svc

    whisper_ok = asr.is_available()
    redis_ok = redis_store.is_available()
    env = settings.environment.lower()
    detailed = env in {"development", "dev", "test"} or settings.debug
    llm_ok = llm.summarizer_available()
    ph_backend = (settings.ph_translate_backend or "auto").strip().lower()
    # Compact three-model map (always present) — see docs/MODELS.md.
    pipeline = {
        "whisper": {
            "role": "asr",
            "owns": "accuracy of what was said",
            "available": whisper_ok,
            "live_model": settings.whisper_live_model,
            "final_model": settings.whisper_final_model,
            "final_backend": settings.whisper_final_backend,
            "default_language": settings.whisper_default_language,
        },
        "mbart_nllb": {
            "role": "translation",
            "owns": "language access",
            "available": llm_ok,
            "ph_translate_backend": ph_backend,
            "nllb_model": settings.nllb_model,
            "mbart_model": settings.mbart_model,
            "mbart_ph_finetuned": bool(
                (settings.mbart_ph_finetuned_model or "").strip()
            ),
        },
        "bart": {
            "role": "summarization",
            "owns": "readable English minutes",
            "available": llm_ok,
            "model": settings.bart_model,
            "allow_fallback": settings.allow_llm_fallback,
        },
    }
    payload = {
        "status": "ok",
        "version": __version__,
        "asr_engine": asr.engine_name(),
        "whisper_available": whisper_ok,
        "redis_available": redis_ok,
        "ffmpeg_available": audio.ffmpeg_available(),
        "llm_available": llm_ok,
        "environment": settings.environment,
        "live_metrics": live_metrics.snapshot(),
        "pipeline": pipeline,
    }
    if not detailed:
        return payload

    # Detailed diagnostics stay available in development / debug only.
    payload.update(
        {
            "whisper_live_model": settings.whisper_live_model,
            "whisper_live_hiligaynon_model": settings.whisper_live_hiligaynon_model
            or None,
            "whisper_live_tagalog_model": settings.whisper_live_tagalog_model or None,
            "whisper_final_model": settings.whisper_final_model,
            "whisper_final_backend": settings.whisper_final_backend,
            "whisper_final_backend_resolved_auto": transcription_svc.resolve_final_backend(
                "auto"
            ),
            "whisper_final_backend_resolved_hil": transcription_svc.resolve_final_backend(
                "hil"
            ),
            "whisper_final_backend_resolved_tl": transcription_svc.resolve_final_backend(
                "tl"
            ),
            "whisper_default_language": settings.whisper_default_language,
            "whisper_hiligaynon_final_language_mode": (
                settings.whisper_hiligaynon_final_language_mode
            ),
            "whisper_hiligaynon_fine_tuned_model": (
                settings.whisper_hiligaynon_fine_tuned_model or None
            ),
            "whisper_hiligaynon_model": settings.whisper_hiligaynon_model or None,
            "whisper_hiligaynon_hf_candidates": transcription_svc.hiligaynon_hf_candidates(),
            "whisper_hiligaynon_primary": transcription_svc.hiligaynon_model_id(),
            "whisper_tagalog_fine_tuned_model": (
                settings.whisper_tagalog_fine_tuned_model or None
            ),
            "whisper_tagalog_model": settings.whisper_tagalog_model or None,
            "whisper_tagalog_hf_candidates": transcription_svc.tagalog_hf_candidates(),
            "whisper_auto_hf_candidates": transcription_svc.auto_hf_candidates(),
            "whisper_tagalog_final_language_mode": settings.whisper_tagalog_final_language_mode,
            "whisper_live_window_seconds": settings.whisper_live_window_seconds,
            "whisper_live_hop_seconds": settings.whisper_live_hop_seconds,
            "whisper_live_backend": settings.whisper_live_backend,
            "rnnt_live": rnnt_svc.status(),
            "redis_url": settings.redis_url if redis_ok else None,
            "max_meeting_hours": settings.max_meeting_hours,
            "redis_audio_ttl_seconds": settings.redis_audio_ttl_seconds,
            "redis_live_buffer_seconds": settings.redis_live_buffer_seconds,
            "redis_rolling_buffer_max_bytes": (
                redis_store.rolling_buffer_max_bytes() if redis_ok else 0
            ),
            "abandoned_session_seconds": settings.abandoned_session_seconds,
            "processing_stale_seconds": settings.processing_stale_seconds,
            "live_asr_max_backlog_windows": settings.live_asr_max_backlog_windows,
            "whisper_final_chunk_seconds": settings.whisper_final_chunk_seconds,
            "whisper_model_cache_size": settings.whisper_model_cache_size,
            "whisper_model_cache": transcription_svc.get_model_cache().stats(),
            "live_caption_prefer_ratio": settings.live_caption_prefer_ratio,
            "live_caption_prefer_min_words": settings.live_caption_prefer_min_words,
        }
    )
    return payload


@app.get("/api/health/transcription", tags=["system"])
def health_transcription():
    """Live-ASR probe for operators. Disabled outside development/debug."""
    import time

    import numpy as np

    from .services import transcription as transcription_svc

    env = settings.environment.lower()
    if env not in {"development", "dev", "test"} and not settings.debug:
        return JSONResponse(
            status_code=404,
            content={"ok": False, "detail": "Not available outside development."},
        )

    if not transcription_svc.is_available():
        return {
            "ok": False,
            "detail": "faster-whisper is not installed",
        }
    # 1s of low-level noise — should return empty (filtered), not crash.
    pcm = np.random.randn(settings.audio_sample_rate).astype(np.float32) * 0.002
    t0 = time.time()
    try:
        segs, detection = transcription_svc.transcribe_live(pcm, "auto")
        return {
            "ok": True,
            "latency_ms": int((time.time() - t0) * 1000),
            "live_model": settings.whisper_live_model,
            "decode_language": "auto",
            "detected_language": detection.language if detection else None,
            "language_detection": detection.as_dict() if detection else None,
            "task": "transcribe",
            "segments": len(segs),
            "backend": settings.whisper_final_backend,
        }
    except Exception as exc:
        return {"ok": False, "detail": str(exc)}


app.include_router(auth.router)
app.include_router(meetings.router)
app.include_router(ai.router)
app.include_router(ws_transcription.router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again."},
    )


def _mount_frontend() -> None:
    """Serve the Vite production build from the same origin as the API.

    Cursor's browser port-forward often reaches :8000 reliably while :5173
    returns ERR_EMPTY_RESPONSE, so shipping the UI through FastAPI avoids that.
    """
    if not (_FRONTEND_DIST.is_dir() and (_FRONTEND_DIST / "index.html").is_file()):
        @app.get("/", tags=["system"])
        def root_api_only():
            return {
                "app": settings.app_name,
                "status": "ok",
                "message": (
                    "API is running, but the frontend build is missing. "
                    "Run: cd frontend && npm run build"
                ),
                "health": "/api/health",
                "docs": "/docs",
            }

        logger.warning("Frontend dist not found at %s", _FRONTEND_DIST)
        return

    assets_dir = _FRONTEND_DIST / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/", include_in_schema=False)
    async def spa_index():
        return FileResponse(_FRONTEND_DIST / "index.html")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        # Never override API / docs / OpenAPI.
        if full_path.startswith(("api/", "docs", "redoc", "openapi.json", "ws/")):
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        candidate = (_FRONTEND_DIST / full_path).resolve()
        try:
            candidate.relative_to(_FRONTEND_DIST.resolve())
        except ValueError:
            return FileResponse(_FRONTEND_DIST / "index.html")
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_FRONTEND_DIST / "index.html")

    logger.info("Serving frontend from %s", _FRONTEND_DIST)


_mount_frontend()
