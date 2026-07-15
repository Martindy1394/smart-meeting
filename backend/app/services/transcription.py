"""Whisper transcription service — the ``TranscribeAudio`` integration.

Implements the two-pass pipeline:

* **Live pass** — fast Whisper model on overlapping 10s windows (5s hop),
  locked to ``language="tl"`` and ``task="transcribe"``.
* **Final pass** — fine-tuned Hiligaynon / Philippine Whisper checkpoint from
  Hugging Face (``WHISPER_HILIGAYNON_MODEL``), also with ``task="transcribe"``
  and ``language="tl"``.

``faster_whisper`` / ``transformers`` are imported lazily so the rest of the
application runs even when model weights are not installed.
"""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass

import numpy as np

from ..config import settings

logger = logging.getLogger("smart_meeting.transcription")

# Soft VAD for the faster-whisper fallback final path.
_FINAL_VAD_PARAMS = {
    "onset": 0.35,
    "offset": 0.25,
    "min_speech_duration_ms": 150,
    "min_silence_duration_ms": 700,
    "speech_pad_ms": 400,
}

# Forced Whisper decode settings (never leave language as None).
_WHISPER_TASK = "transcribe"
_WHISPER_LANGUAGE = "tl"

_INITIAL_PROMPT = (
    "Meeting minutes in Hiligaynon, Filipino, or English. "
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
    """Lazily loads and caches Whisper backends, thread-safely."""

    def __init__(self) -> None:
        self._fw_models: dict[str, object] = {}
        self._hf_pipelines: dict[str, object] = {}
        self._lock = threading.Lock()

    def get_faster_whisper(self, model_size: str):
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise TranscriptionUnavailable(
                "faster-whisper is not installed. Install backend ML deps: "
                "pip install -r requirements-ml.txt"
            ) from exc

        with self._lock:
            if model_size not in self._fw_models:
                logger.info(
                    "Loading faster-whisper model '%s' (device=%s)",
                    model_size,
                    settings.whisper_device,
                )
                self._fw_models[model_size] = WhisperModel(
                    model_size,
                    device=settings.whisper_device,
                    compute_type=settings.whisper_compute_type,
                )
            return self._fw_models[model_size]

    def get_hf_pipeline(self, model_id: str):
        """Load a Hugging Face fine-tuned Whisper ASR pipeline."""
        try:
            import torch  # type: ignore
            from transformers import pipeline  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise TranscriptionUnavailable(
                "transformers/torch are not installed. Install backend ML deps: "
                "pip install -r requirements-ml.txt"
            ) from exc

        with self._lock:
            if model_id not in self._hf_pipelines:
                device = 0 if settings.whisper_device == "cuda" and torch.cuda.is_available() else -1
                dtype = torch.float16 if device >= 0 else torch.float32
                logger.info(
                    "Loading fine-tuned Whisper ASR '%s' via transformers (device=%s)",
                    model_id,
                    "cuda" if device >= 0 else "cpu",
                )
                self._hf_pipelines[model_id] = pipeline(
                    "automatic-speech-recognition",
                    model=model_id,
                    device=device,
                    torch_dtype=dtype,
                    chunk_length_s=30,
                    stride_length_s=5,
                    return_timestamps=True,
                )
            return self._hf_pipelines[model_id]


_cache = _ModelCache()


def is_available() -> bool:
    try:
        import faster_whisper  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def hiligaynon_model_id() -> str:
    """Hugging Face / local id used for the final Stop-Recording pass."""
    return (settings.whisper_hiligaynon_model or settings.whisper_final_model).strip()


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


def _forced_language(_requested: str | None) -> str:
    """Always return Tagalog (``tl``) — never leave Whisper language as None."""
    return _WHISPER_LANGUAGE


def transcribe_live(pcm: np.ndarray, language: str | None) -> list[Segment]:
    """Low-latency transcription of a live audio window.

    Uses overlapping windows upstream; decode is locked to Tagalog transcription.
    """
    if pcm is None or len(pcm) == 0:
        return []

    model = _cache.get_faster_whisper(settings.whisper_live_model)
    lang = _forced_language(language)
    segments, _info = model.transcribe(
        pcm,
        language=lang,
        task=_WHISPER_TASK,
        beam_size=3,
        best_of=3,
        temperature=0.0,
        vad_filter=False,
        condition_on_previous_text=False,
        without_timestamps=False,
        no_speech_threshold=0.2,
        compression_ratio_threshold=3.2,
        log_prob_threshold=-1.5,
        initial_prompt=_INITIAL_PROMPT,
    )
    out: list[Segment] = []
    for s in segments:
        text = _collapse_hallucinations((s.text or "").strip())
        if text and any(ch.isalnum() for ch in text):
            out.append(Segment(text=text, start=s.start, end=s.end))
    return out


def merge_live_caption(previous: str, window_text: str) -> str:
    """Merge an overlapping-window transcript into the running live caption."""
    prev = re.sub(r"\s+", " ", (previous or "").strip())
    cur = re.sub(r"\s+", " ", (window_text or "").strip())
    if not cur:
        return prev
    if not prev:
        return cur
    if cur.lower() in prev.lower():
        return prev

    prev_tokens = prev.split()
    cur_tokens = cur.split()
    max_overlap = min(len(prev_tokens), len(cur_tokens), 40)
    overlap = 0
    for size in range(max_overlap, 0, -1):
        left = [t.lower() for t in prev_tokens[-size:]]
        right = [t.lower() for t in cur_tokens[:size]]
        if left == right:
            overlap = size
            break

    if overlap == 0:
        prev_l = prev.lower()
        cur_l = cur.lower()
        max_check = min(len(prev_l), len(cur_l), 80)
        for n in range(max_check, 2, -1):
            if prev_l.endswith(cur_l[:n]):
                suffix = cur[n:].lstrip(" ,.;:-")
                if not suffix:
                    return prev
                joiner = "" if prev.endswith((" ", "\n")) or suffix[:1].isspace() else " "
                return f"{prev}{joiner}{suffix}".strip()
        return f"{prev} {cur}".strip()

    merged = prev_tokens + cur_tokens[overlap:]
    return " ".join(merged).strip()


def _segments_to_list(segments) -> list[Segment]:
    out: list[Segment] = []
    for s in segments:
        text = _collapse_hallucinations((s.text or "").strip())
        if text:
            out.append(Segment(text=text, start=s.start, end=s.end))
    return out


def _audio_to_float32(audio_source) -> np.ndarray:
    if isinstance(audio_source, np.ndarray):
        return audio_source.astype(np.float32, copy=False)
    from . import audio as audio_svc

    return audio_svc.load_audio_float32(str(audio_source))


def _transcribe_final_hf(audio_source, language: str | None) -> list[Segment]:
    """Final pass with a Hugging Face fine-tuned Whisper checkpoint."""
    model_id = hiligaynon_model_id()
    pipe = _cache.get_hf_pipeline(model_id)
    samples = _audio_to_float32(audio_source)
    if samples.size == 0:
        return []

    lang = _forced_language(language)
    logger.info(
        "Final Whisper ASR with fine-tuned model '%s' (language=%s, task=%s)",
        model_id,
        lang,
        _WHISPER_TASK,
    )
    result = pipe(
        {"array": samples, "sampling_rate": int(settings.audio_sample_rate)},
        generate_kwargs={
            "language": lang,
            "task": _WHISPER_TASK,
        },
        return_timestamps=True,
    )

    chunks = result.get("chunks") if isinstance(result, dict) else None
    if chunks:
        out: list[Segment] = []
        for i, chunk in enumerate(chunks):
            text = _collapse_hallucinations((chunk.get("text") or "").strip())
            if not text:
                continue
            ts = chunk.get("timestamp") or (None, None)
            start = float(ts[0] or 0.0)
            end = float(ts[1] if ts[1] is not None else start)
            out.append(Segment(text=text, start=start, end=end))
        if out:
            return out

    text = ""
    if isinstance(result, dict):
        text = _collapse_hallucinations((result.get("text") or "").strip())
    elif isinstance(result, str):
        text = _collapse_hallucinations(result.strip())
    if not text:
        return []
    duration = float(samples.size) / float(settings.audio_sample_rate)
    return [Segment(text=text, start=0.0, end=duration)]


def _transcribe_final_faster_whisper(audio_source, language: str | None) -> list[Segment]:
    """Fallback final pass via faster-whisper (stock or converted model id)."""
    model = _cache.get_faster_whisper(settings.whisper_final_model)
    lang = _forced_language(language)
    segments, info = model.transcribe(
        audio_source if isinstance(audio_source, str) else _audio_to_float32(audio_source),
        language=lang,
        task=_WHISPER_TASK,
        beam_size=5,
        best_of=5,
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        vad_filter=True,
        vad_parameters=_FINAL_VAD_PARAMS,
        condition_on_previous_text=False,
        without_timestamps=False,
        initial_prompt=_INITIAL_PROMPT,
        no_speech_threshold=0.55,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
    )
    detected = getattr(info, "language", None)
    if detected:
        logger.info("faster-whisper final language reported: %s", detected)
    return _segments_to_list(segments)


def _looks_like_hf_repo(model_id: str) -> bool:
    mid = (model_id or "").strip()
    return "/" in mid or mid.startswith(".") or mid.startswith("/")


def transcribe_final(audio_source, language: str | None) -> list[Segment]:
    """Full-accuracy transcription for Stop Recording / re-transcribe / upload.

    Prefer the configured fine-tuned Hiligaynon (or PH-dialect) Hugging Face
    Whisper model. Fall back to faster-whisper if that checkpoint cannot load.
    Always forces ``language="tl"`` and ``task="transcribe"``.
    """
    model_id = hiligaynon_model_id()
    if _looks_like_hf_repo(model_id):
        try:
            return _transcribe_final_hf(audio_source, language)
        except TranscriptionUnavailable:
            raise
        except Exception as exc:
            logger.exception(
                "Fine-tuned Hiligaynon Whisper '%s' failed (%s); "
                "falling back to faster-whisper '%s'.",
                model_id,
                exc,
                settings.whisper_final_model,
            )

    return _transcribe_final_faster_whisper(audio_source, language)
