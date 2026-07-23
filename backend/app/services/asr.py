"""Whisper + optional NeMo RNN-T ASR for meeting audio.

This is the single integration surface for turning audio into text:

* Live PCM windows → ``transcribe_pcm`` (faster-whisper, or NeMo RNN-T when enabled)
* Saved / uploaded WAV files → ``transcribe_file`` (full-accuracy Whisper pass)
* Meeting pipeline → persist segments + transcript

Live captions prefer FastConformer RNN-T for **Tagalog** meetings when
``WHISPER_LIVE_BACKEND=auto|rnnt`` and NeMo is installed. Hiligaynon-biased
``auto`` meetings stay on Whisper. Final pass stays on Whisper / PH-medium
(see ``transcription.py``).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from . import audio, transcription

logger = logging.getLogger("smart_meeting.asr")

# Public alias so callers can treat ASR errors uniformly.
ASRUnavailable = transcription.TranscriptionUnavailable
Segment = transcription.Segment
LanguageDetection = transcription.LanguageDetection


@dataclass
class ASRResult:
    """Normalized ASR output for a processed audio source."""

    text: str
    segments: list[Segment]
    engine: str = "whisper"
    language: str | None = None
    language_confidence: float | None = None
    language_detected_by: str = ""


def is_available() -> bool:
    """True when at least one live ASR backend can be loaded."""
    if transcription.is_available():
        return True
    try:
        from . import rnnt

        return rnnt.is_available()
    except Exception:
        return False


def engine_name() -> str:
    try:
        from . import rnnt

        if rnnt.should_use_rnnt_live("tl") and rnnt.is_available():
            return "rnnt+whisper"
    except Exception:
        pass
    return "whisper" if transcription.is_available() else "unavailable"


def transcribe_pcm(pcm: np.ndarray, language: str | None = None, *, live: bool = False) -> ASRResult:
    """Run Whisper ASR on a float32 PCM buffer.

    ``live=True`` uses the low-latency live model; otherwise the full-accuracy
    finalization model is used (same quality as file transcription).
    """
    if not is_available():
        raise ASRUnavailable(
            "Whisper ASR is not installed. Install backend ML deps: "
            "pip install -r requirements-ml.txt"
        )
    if pcm is None or len(pcm) == 0:
        return ASRResult(text="", segments=[], engine="whisper", language=language)

    if live:
        segments, detection = transcription.transcribe_live(pcm, language)
    else:
        segments, detection = transcription.transcribe_final(pcm, language)

    text = " ".join(s.text for s in segments).strip()
    resolved = (detection.language if detection else None) or language
    detected_by = (detection.detected_by if detection else "") or ""
    engine = "rnnt" if detected_by == "rnnt" else "whisper"
    return ASRResult(
        text=text,
        segments=segments,
        engine=engine,
        language=resolved,
        language_confidence=detection.confidence if detection else None,
        language_detected_by=detected_by,
    )


def transcribe_file(path: str, language: str | None = None) -> ASRResult:
    """Run full-accuracy Whisper ASR on a stored audio file (WAV path)."""
    if not is_available():
        raise ASRUnavailable(
            "Whisper ASR is not installed. Install backend ML deps: "
            "pip install -r requirements-ml.txt"
        )
    samples = audio.load_audio_float32(path)
    return transcribe_pcm(samples, language, live=False)


def transcribe_pcm_bytes(pcm_bytes: bytes, language: str | None = None, *, live: bool = False) -> ASRResult:
    """Convenience: int16 LE PCM bytes → Whisper ASR."""
    samples = audio.pcm16_to_float32(pcm_bytes)
    return transcribe_pcm(samples, language, live=live)


def merge_live_caption(
    previous: str,
    window_text: str,
    *,
    previous_window: str | None = None,
) -> str:
    """Merge overlapping live Whisper windows into a continuous caption."""
    return transcription.merge_live_caption(
        previous, window_text, previous_window=previous_window
    )


def persist_transcript(db, meeting, result: ASRResult) -> None:
    """Write Whisper ASR segments + full text onto a meeting row.

    Clears stale summary/translation because they were derived from older text.
    """
    from ..models import TranscriptSegment

    db.query(TranscriptSegment).filter(
        TranscriptSegment.meeting_id == meeting.id
    ).delete()
    for i, seg in enumerate(result.segments):
        db.add(
            TranscriptSegment(
                meeting_id=meeting.id,
                kind="final",
                text=seg.text,
                start_time=seg.start,
                end_time=seg.end,
                seq=i,
            )
        )
    meeting.final_transcript = result.text
    # Final safety net: collapse any residual Whisper repetition loops.
    try:
        from .transcription import _collapse_hallucinations

        meeting.final_transcript = _collapse_hallucinations(result.text)
    except Exception:
        pass
    meeting.summary = ""
    meeting.summary_format = ""
    meeting.translation = ""
    meeting.translation_language = ""
    # Spoken language is not user-selected. Always keep ``auto`` so every pass
    # uses Hiligaynon-biased automatic detection (``effective_asr_language``).
    # Whisper's guessed code lives in language_confidence / language_detected_by.
    meeting.language = "auto"
    meeting.language_confidence = result.language_confidence
    meeting.language_detected_by = (result.language_detected_by or "").strip()
    meeting.status = "finalized"
