"""Whisper ASR — automatic speech recognition for meeting audio.

This is the single integration surface for turning audio into text:

* Live PCM windows → ``transcribe_pcm`` (fast captions)
* Saved / uploaded WAV files → ``transcribe_file`` (full-accuracy pass)
* Meeting pipeline → ``process_meeting_audio`` (persist segments + transcript)

Live captions use faster-whisper (optional CT2 Hiligaynon fine-tune). Final
pass prefers a fine-tuned Hiligaynon HF checkpoint when configured, then
falls back to faster-whisper (see ``transcription.py``).
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
    """Normalized Whisper ASR output for a processed audio source."""

    text: str
    segments: list[Segment]
    engine: str = "whisper"
    language: str | None = None
    language_confidence: float | None = None
    language_detected_by: str = ""


def is_available() -> bool:
    """True when the Whisper ASR backend can be loaded."""
    return transcription.is_available()


def engine_name() -> str:
    return "whisper" if is_available() else "unavailable"


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
    return ASRResult(
        text=text,
        segments=segments,
        engine="whisper",
        language=resolved,
        language_confidence=detection.confidence if detection else None,
        language_detected_by=(detection.detected_by if detection else "") or "",
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
    # Persist Whisper-detected language + confidence metadata.
    # Do not overwrite an explicit user label (hil/tl/en) — that label drives
    # Hiligaynon/Tagalog prompt bias on the next re-transcribe.
    detected = (result.language or "").strip().lower()
    prev_lang = (meeting.language or "auto").strip().lower()
    if prev_lang in {"", "auto", "detect", "none"} and detected and detected not in {
        "auto",
        "detect",
        "none",
    }:
        meeting.language = detected
    meeting.language_confidence = result.language_confidence
    meeting.language_detected_by = (result.language_detected_by or "").strip()
    meeting.status = "finalized"
