"""FastAPI application entry point for Smart Meeting."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import __version__
from .config import settings
from .database import init_db
from .routers import ai, auth, meetings
from .services import llm, transcription
from .ws import transcription as ws_transcription

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("smart_meeting")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s v%s (%s)", settings.app_name, __version__, settings.environment)
    init_db()
    yield
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version=__version__,
    description="Minute-making platform: live Whisper transcription, BART "
    "summarization, and mBART translation.",
    lifespan=lifespan,
)

# CORS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["system"])
def root():
    return {
        "app": settings.app_name,
        "status": "ok",
        "message": "API is running. Open the frontend at http://127.0.0.1:5173/",
        "health": "/api/health",
        "docs": "/docs",
    }


@app.get("/api/health", tags=["system"])
def health():
    return {
        "status": "ok",
        "version": __version__,
        "whisper_available": transcription.is_available(),
        "llm_available": llm.summarizer_available(),
        "environment": settings.environment,
    }


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
