"""Whisper transcription service — the ``TranscribeAudio`` integration.

Implements the two-pass pipeline described in the specification:

* **Live pass** — a fast, low-latency model (``base``/``tiny``) with greedy
  decoding, used for word-by-word captions during recording.
* **Final pass** — the full-accuracy model (``large-v3``) with beam search
  (``beam_size=5``, ``best_of=5``), VAD filtering, ``temperature=0.0`` and
  ``condition_on_previous_text=True`` so the finalized transcript reads as one
  coherent document.

The Hiligaynon dialect is requested via ``language="hil"``.  Because upstream
Whisper builds do not always ship a ``hil`` language token, we validate the
requested code against the model's supported set and fall back to the closest
Philippine language (Tagalog) when necessary, logging a warning.

``faster_whisper`` is imported lazily so the rest of the application runs even
when the (multi-gigabyte) model weights / package are not installed.
"""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass

import numpy as np

from ..config import settings

logger = logging.getLogger("smart_meeting.transcription")

# Closest supported language when the requested dialect is unavailable.
_LANGUAGE_FALLBACK = {"hil": "tl"}

_JUNK_CAPTION_RE = re.compile(
    r"^[\s\.\,\!\?…\-–—\"'“”‘’]*$"
    r"|^(thanks for watching|thank you for watching|subscribe|bye\.?)$",
    re.IGNORECASE,
)


def _is_junk_caption(text: str) -> bool:
    if not text:
        return True
    if _JUNK_CAPTION_RE.match(text.strip()):
        return True
    letters = sum(ch.isalpha() for ch in text)
    return letters < 2


class TranscriptionUnavailable(RuntimeError):
    """Raised when the Whisper backend cannot be loaded."""


@dataclass
class Segment:
    text: str
    start: float
    end: float


class _ModelCache:
    """Lazily loads and caches faster-whisper models by size, thread-safely."""

    def __init__(self) -> None:
        self._models: dict[str, object] = {}
        self._lock = threading.Lock()
        self._supported_languages: set[str] | None = None

    def _load(self, model_size: str):
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional dep
            raise TranscriptionUnavailable(
                "faster-whisper is not installed. Install backend ML deps: "
                "pip install -r requirements-ml.txt"
            ) from exc

        logger.info("Loading Whisper model '%s' (device=%s)", model_size, settings.whisper_device)
        model = WhisperModel(
            model_size,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
        # Capture the model's supported languages for validation.
        try:
            from faster_whisper.tokenizer import _LANGUAGE_CODES  # type: ignore

            self._supported_languages = set(_LANGUAGE_CODES)
        except Exception:
            self._supported_languages = None
        return model

    def get(self, model_size: str):
        with self._lock:
            if model_size not in self._models:
                self._models[model_size] = self._load(model_size)
            return self._models[model_size]

    def resolve_language(self, requested: str | None) -> str | None:
        if not requested:
            return None
        supported = self._supported_languages
        if supported is None:
            # Unknown support set — trust the caller but map known problem codes.
            return _LANGUAGE_FALLBACK.get(requested, requested)
        if requested in supported:
            return requested
        fallback = _LANGUAGE_FALLBACK.get(requested)
        if fallback and fallback in supported:
            logger.warning(
                "Whisper does not support language '%s'; falling back to '%s'.",
                requested,
                fallback,
            )
            return fallback
        logger.warning(
            "Whisper does not support language '%s'; using auto-detection.", requested
        )
        return None


_cache = _ModelCache()


def is_available() -> bool:
    try:
        import faster_whisper  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def transcribe_live(pcm: np.ndarray, language: str | None) -> list[Segment]:
    """Fast, low-latency transcription of a short audio window (live captions).

    ``pcm`` must be mono float32 in [-1, 1] at ``settings.audio_sample_rate``.
    Live mode disables VAD (too aggressive on short/quiet windows) and
    peak-normalizes the chunk first so captions keep flowing.
    """
    from . import audio as audio_util

    model = _cache.get(settings.whisper_live_model)
    lang = _cache.resolve_language(language or settings.whisper_default_language)
    samples = audio_util.normalize_audio(np.asarray(pcm, dtype=np.float32))
    if audio_util.rms_level(samples) < 1e-4:
        return []
    segments, _info = model.transcribe(
        samples,
        language=lang,
        beam_size=1,
        best_of=1,
        temperature=0.0,
        # Short live windows + quiet mics: VAD often wipes the entire chunk.
        vad_filter=False,
        condition_on_previous_text=False,
        without_timestamps=True,
        # Suppress hallucinations on near-silence / noise.
        no_speech_threshold=0.5,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-0.8,
    )
    out: list[Segment] = []
    for s in segments:
        text = (s.text or "").strip()
        if _is_junk_caption(text):
            continue
        out.append(
            Segment(
                text=text,
                start=float(getattr(s, "start", 0.0) or 0.0),
                end=float(getattr(s, "end", 0.0) or 0.0),
            )
        )
    return out


def transcribe_final(audio_source, language: str | None) -> list[Segment]:
    """Full-accuracy transcription of the complete recording (finalization pass).

    ``audio_source`` may be a file path (str) or a float32 numpy array.
    """
    from . import audio as audio_util

    model = _cache.get(settings.whisper_final_model)
    lang = _cache.resolve_language(language or settings.whisper_default_language)

    if isinstance(audio_source, np.ndarray):
        audio_source = audio_util.normalize_audio(np.asarray(audio_source, dtype=np.float32))

    # Correct VadOptions keys for this faster-whisper version: onset/offset
    # (not "threshold"). Soft settings keep quieter speech after normalization.
    vad_parameters = {
        "onset": 0.35,
        "offset": 0.25,
        "min_speech_duration_ms": 150,
        "min_silence_duration_ms": 400,
        "speech_pad_ms": 400,
    }

    try:
        segments, _info = model.transcribe(
            audio_source,
            language=lang,
            beam_size=5,
            best_of=5,
            temperature=0.0,
            vad_filter=True,
            vad_parameters=vad_parameters,
            condition_on_previous_text=True,
            without_timestamps=False,
        )
        out = [
            Segment(text=s.text.strip(), start=s.start, end=s.end)
            for s in segments
            if (s.text or "").strip()
        ]
    except TypeError as exc:
        # Older/newer VadOptions mismatch — retry without custom params.
        logger.warning("VAD params unsupported (%s); retrying with defaults.", exc)
        segments, _info = model.transcribe(
            audio_source,
            language=lang,
            beam_size=5,
            best_of=5,
            temperature=0.0,
            vad_filter=True,
            condition_on_previous_text=True,
            without_timestamps=False,
        )
        out = [
            Segment(text=s.text.strip(), start=s.start, end=s.end)
            for s in segments
            if (s.text or "").strip()
        ]

    # If VAD wiped everything, retry without VAD so we still return a transcript.
    if not out and isinstance(audio_source, np.ndarray) and audio_util.rms_level(audio_source) > 1e-4:
        logger.warning("Final VAD produced no speech — retrying without VAD filter.")
        segments, _info = model.transcribe(
            audio_source,
            language=lang,
            beam_size=5,
            best_of=5,
            temperature=0.0,
            vad_filter=False,
            condition_on_previous_text=True,
            without_timestamps=False,
        )
        out = [
            Segment(text=s.text.strip(), start=s.start, end=s.end)
            for s in segments
            if (s.text or "").strip() and not _is_junk_caption(s.text.strip())
        ]
    return out
