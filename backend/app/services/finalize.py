"""Shared finalization for WebSocket stop and REST stop.

Reads authoritative PCM from disk (with Redis rolling-buffer fallback), runs the
full-accuracy Whisper pass, persists transcript + WAV, and clears live buffers.
"""
from __future__ import annotations

import logging

from ..config import settings
from ..database import SessionLocal
from ..models import Meeting
from . import asr, audio, live_metrics, redis_store

logger = logging.getLogger("smart_meeting.finalize")


def load_recording_pcm(meeting_id: str) -> bytes:
    """Prefer on-disk PCM; fall back to Redis rolling buffer (legacy/short)."""
    pcm = audio.read_raw_pcm(meeting_id)
    if pcm:
        return pcm
    return redis_store.get_pcm(meeting_id)


def finalize_meeting_recording(
    meeting_id: str,
    live_caption: str = "",
    *,
    language: str | None = None,
) -> dict:
    """Run final ASR and persist. Safe to call from WS stop or REST stop."""
    db = SessionLocal()
    try:
        meeting = db.get(Meeting, meeting_id)
        if meeting is None:
            live_metrics.record_finalize(False)
            return {"ok": False, "message": "Meeting not found."}

        if meeting.status in ("finalized", "processing"):
            live_metrics.record_finalize(True)
            return {
                "ok": True,
                "already_done": True,
                "text": meeting.final_transcript or "",
                "segments": [],
                "engine": "already-finalized",
                "status": meeting.status,
            }

        if not live_caption:
            meta = redis_store.get_session_meta(meeting_id)
            live_caption = meta.get("live_caption") or ""

        lang = language or meeting.language or settings.whisper_default_language
        pcm_bytes = load_recording_pcm(meeting_id)

        if len(pcm_bytes) < 2:
            fallback = (live_caption or "").strip()
            meeting.status = "finalized"
            if fallback and not (meeting.final_transcript or "").strip():
                meeting.final_transcript = fallback
            db.commit()
            redis_store.clear_meeting_audio(meeting_id, keep_wav=True)
            audio.delete_raw_pcm(meeting_id)
            live_metrics.record_finalize(True)
            return {
                "ok": True,
                "text": fallback,
                "segments": (
                    [{"text": fallback, "start": 0.0, "end": 0.0}] if fallback else []
                ),
                "engine": "empty-recording",
            }

        audio_path = audio.save_wav(meeting_id, pcm_bytes)
        meeting.audio_path = audio_path
        meeting.duration_seconds = audio.wav_duration_seconds(pcm_bytes)
        meeting.status = "processing"
        db.commit()

        try:
            result = asr.transcribe_pcm_bytes(pcm_bytes, lang, live=False)
        except asr.ASRUnavailable as exc:
            live = (live_caption or "").strip()
            if live:
                result = asr.ASRResult(
                    text=live,
                    segments=[
                        asr.Segment(
                            text=live, start=0.0, end=meeting.duration_seconds or 0.0
                        )
                    ],
                    engine="whisper-live-fallback",
                    language=lang,
                )
            else:
                meeting.status = "failed"
                db.commit()
                live_metrics.record_finalize(False)
                return {"ok": False, "message": str(exc)}

        live = (live_caption or "").strip()
        final_text = (result.text or "").strip()
        live_words = len(live.split()) if live else 0
        final_words = len(final_text.split()) if final_text else 0
        prefer_ratio = min(1.0, max(0.0, float(settings.live_caption_prefer_ratio)))
        min_live_words = max(1, int(settings.live_caption_prefer_min_words))
        final_threshold = max(1, int(live_words * prefer_ratio))
        # Only prefer live when the final pass is drastically shorter — never
        # replace a solid final transcript with a partial live merge.
        prefer_live = (
            live_words >= min_live_words
            and final_words < final_threshold
            and final_words < max(8, int(live_words * 0.35))
        )
        logger.info(
            "asr.finalize_choice meeting=%s live_words=%d final_words=%d "
            "prefer_ratio=%.2f min_live_words=%d threshold=%d prefer_live=%s",
            meeting_id,
            live_words,
            final_words,
            prefer_ratio,
            min_live_words,
            final_threshold,
            prefer_live,
        )
        if prefer_live:
            result = asr.ASRResult(
                text=live,
                segments=[
                    asr.Segment(
                        text=live, start=0.0, end=meeting.duration_seconds or 0.0
                    )
                ],
                engine=f"{result.engine}+live-caption",
                language=lang,
            )

        asr.persist_transcript(db, meeting, result)
        db.commit()

        redis_store.clear_meeting_audio(meeting_id, keep_wav=True)
        audio.delete_raw_pcm(meeting_id)
        live_metrics.record_finalize(True)

        return {
            "ok": True,
            "text": result.text,
            "segments": [
                {"text": s.text, "start": s.start, "end": s.end}
                for s in result.segments
            ],
            "engine": result.engine,
            "status": meeting.status,
        }
    except Exception as exc:
        logger.exception("Finalize failed for meeting %s", meeting_id)
        live_metrics.record_finalize(False)
        try:
            meeting = db.get(Meeting, meeting_id)
            if meeting and meeting.status == "processing":
                meeting.status = "failed"
                db.commit()
        except Exception:
            db.rollback()
        return {"ok": False, "message": str(exc)}
    finally:
        db.close()
