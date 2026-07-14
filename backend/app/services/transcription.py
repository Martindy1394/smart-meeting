"""Whisper transcription service — the ``TranscribeAudio`` integration.

Implements the two-pass pipeline described in the specification:

* **Live pass** — a fast, low-latency model (``base``/``tiny``) with greedy
  decoding, used for word-by-word captions during recording.
* **Final pass** — the full-accuracy model with beam search, soft VAD,
  temperature fallback, and anti-hallucination settings so the finalized
  transcript reads as one coherent document.

Hiligaynon (``hil``) is not a Whisper language token. Forcing Tagalog
(``tl``) caused severe hallucination loops on English / mixed Philippine
speech (common in meetings). We therefore use **auto language detection**
for Hiligaynon instead of a hard Tagalog lock.

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

# Languages Whisper cannot lock to — use auto-detect instead of a wrong force.
_AUTO_DETECT_LANGUAGES = frozenset({"hil"})

# Soft VAD: keep more quiet speech; pad edges so words aren't clipped.
# faster-whisper VadOptions uses onset/offset (not "threshold").
_FINAL_VAD_PARAMS = {
    "onset": 0.35,
    "offset": 0.25,
    "min_speech_duration_ms": 150,
    "min_silence_duration_ms": 700,
    "speech_pad_ms": 400,
}

# Philippine-meeting hint so auto-detect prefers English / Filipino lexicon
# without forcing Tagalog decoding.
_INITIAL_PROMPT = (
    "Meeting minutes. Speakers may use English, Filipino, or Hiligaynon. "
    "Transcribe the spoken words accurately."
)


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
        """Return a Whisper language code, or ``None`` for auto-detect."""
        if not requested:
            return None
        requested = requested.strip().lower()
        if requested in _AUTO_DETECT_LANGUAGES:
            logger.info(
                "Language '%s' is not supported by Whisper; using auto-detection "
                "(avoids forced-Tagalog hallucination on English/mixed speech).",
                requested,
            )
            return None
        supported = self._supported_languages
        if supported is None:
            return requested
        if requested in supported:
            return requested
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


_REPEAT_WORD_RE = re.compile(r"\b(\w+)(?:\s+\1){3,}\b", re.IGNORECASE)
_STUTTER_RE = re.compile(r"([A-Za-zÀ-ÿ])(?:-\1){3,}", re.IGNORECASE)


def _collapse_hallucinations(text: str) -> str:
    """Collapse obvious Whisper repetition loops (e.g. 'mic mic mic…')."""
    prev = None
    while prev != text:
        prev = text
        text = _REPEAT_WORD_RE.sub(r"\1", text)
        text = _STUTTER_RE.sub(r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def transcribe_live(pcm: np.ndarray, language: str | None) -> list[Segment]:
    """Fast, low-latency transcription of a short audio window (live captions).

    ``pcm`` must be mono float32 in [-1, 1] at ``settings.audio_sample_rate``.
    """
    model = _cache.get(settings.whisper_live_model)
    lang = _cache.resolve_language(language or settings.whisper_default_language)
    segments, _info = model.transcribe(
        pcm,
        language=lang,
        beam_size=1,
        best_of=1,
        temperature=0.0,
        # Live windows are short; VAD often drops the whole chunk.
        vad_filter=False,
        condition_on_previous_text=False,
        without_timestamps=False,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
    )
    out: list[Segment] = []
    for s in segments:
        text = _collapse_hallucinations((s.text or "").strip())
        if text and any(ch.isalpha() for ch in text):
            out.append(Segment(text=text, start=s.start, end=s.end))
    return out


def _run_final(model, audio_source, language: str | None, *, use_prompt: bool):
    return model.transcribe(
        audio_source,
        language=language,
        beam_size=5,
        best_of=5,
        # Retry with higher temperatures when compression/log-prob checks fail.
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        vad_filter=True,
        vad_parameters=_FINAL_VAD_PARAMS,
        # Avoid prompt-conditioning loops ("word word word…") on long audio.
        condition_on_previous_text=False,
        without_timestamps=False,
        initial_prompt=_INITIAL_PROMPT if use_prompt else None,
        no_speech_threshold=0.55,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
    )


def _segments_to_list(segments) -> list[Segment]:
    out: list[Segment] = []
    for s in segments:
        text = _collapse_hallucinations((s.text or "").strip())
        if text:
            out.append(Segment(text=text, start=s.start, end=s.end))
    return out


# Languages that are plausible for Philippine meetings / code-switching.
_PLAUSIBLE_LANGS = frozenset({"en", "tl", "es", "zh", "ja", "ko", "id", "ms"})


def transcribe_final(audio_source, language: str | None) -> list[Segment]:
    """Full-accuracy transcription of the complete recording (finalization pass).

    ``audio_source`` may be a file path (str) or a float32 numpy array.
    """
    model = _cache.get(settings.whisper_final_model)
    requested = (language or settings.whisper_default_language or "").strip().lower()
    lang = _cache.resolve_language(requested or None)

    segments, info = _run_final(model, audio_source, lang, use_prompt=(lang is None))
    detected = getattr(info, "language", None)
    prob = float(getattr(info, "language_probability", 0.0) or 0.0)
    if detected:
        logger.info(
            "Final transcription language detected: %s (p=%.2f)", detected, prob
        )

    # When Hiligaynon forced auto-detect and confidence is poor / implausible
    # (e.g. "jw" on mic noise), retry locked to English — common in PH meetings.
    if (
        requested in _AUTO_DETECT_LANGUAGES
        and lang is None
        and (prob < 0.55 or (detected and detected not in _PLAUSIBLE_LANGS))
    ):
        logger.info(
            "Low-confidence detect (%s p=%.2f); retrying final pass as English.",
            detected,
            prob,
        )
        segments, info = _run_final(model, audio_source, "en", use_prompt=False)
        detected = getattr(info, "language", None)
        logger.info("English retry language: %s", detected)

    return _segments_to_list(segments)
