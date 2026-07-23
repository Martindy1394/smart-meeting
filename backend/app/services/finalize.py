"""Shared finalization for WebSocket stop and REST stop.

Reads authoritative PCM from disk (with Redis rolling-buffer fallback), runs the
full-accuracy Whisper pass, persists transcript + WAV, and clears live buffers.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import and_, update

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


def _aware(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def is_processing_stale(meeting: Meeting, *, now: datetime | None = None) -> bool:
    """True when a ``processing`` meeting has been stuck longer than the lease."""
    if meeting.status != "processing":
        return False
    stamp = _aware(meeting.updated_at) or _aware(meeting.created_at)
    if stamp is None:
        return True
    now = now or datetime.now(timezone.utc)
    age = (now - stamp).total_seconds()
    return age >= max(60, int(settings.processing_stale_seconds))


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

        if meeting.status == "finalized":
            live_metrics.record_finalize(True)
            return {
                "ok": True,
                "already_done": True,
                "text": meeting.final_transcript or "",
                "segments": [],
                "engine": "already-finalized",
                "status": meeting.status,
            }

        if meeting.status == "processing" and not is_processing_stale(meeting):
            # Another worker owns the ASR lease — do not pretend success.
            live_metrics.record_finalize(True)
            return {
                "ok": True,
                "already_done": True,
                "in_progress": True,
                "text": meeting.final_transcript or "",
                "segments": [],
                "engine": "processing",
                "status": meeting.status,
                "message": "Transcription is already in progress.",
            }

        if not live_caption:
            meta = redis_store.get_session_meta(meeting_id)
            live_caption = meta.get("live_caption") or ""

        # Resolve auto/empty → Hiligaynon by default (Whisper has no hil token).
        from . import transcription as transcription_svc

        lang = transcription_svc.effective_asr_language(
            language or (meeting.language if meeting else None)
        )
        pcm_bytes = load_recording_pcm(meeting_id)

        if len(pcm_bytes) < 2:
            fallback = (live_caption or "").strip()
            meeting.status = "finalized"
            meeting.updated_at = datetime.now(timezone.utc)
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
        duration = audio.wav_duration_seconds(pcm_bytes)
        # Free the large PCM buffer before Whisper loads float32 from the WAV.
        del pcm_bytes

        # Atomic lease: one finalize/reclaim may enter (or refresh) processing.
        now = datetime.now(timezone.utc)
        prior_status = meeting.status
        prior_updated = meeting.updated_at
        if prior_status == "processing":
            claim_filter = and_(
                Meeting.id == meeting_id,
                Meeting.status == "processing",
                Meeting.updated_at == prior_updated,
            )
        else:
            claim_filter = and_(
                Meeting.id == meeting_id,
                Meeting.status.in_(("recording", "failed")),
            )
        claim = db.execute(
            update(Meeting)
            .where(claim_filter)
            .values(
                audio_path=audio_path,
                duration_seconds=duration,
                status="processing",
                updated_at=now,
            )
        )
        db.commit()
        if claim.rowcount == 0:
            meeting = db.get(Meeting, meeting_id)
            status = meeting.status if meeting else "unknown"
            live_metrics.record_finalize(True)
            return {
                "ok": True,
                "already_done": True,
                "in_progress": status == "processing",
                "text": (meeting.final_transcript if meeting else "") or "",
                "segments": [],
                "engine": "lease-lost",
                "status": status,
            }

        meeting = db.get(Meeting, meeting_id)
        assert meeting is not None

        try:
            # Prefer file path so we do not keep a second full PCM copy in RAM.
            result = asr.transcribe_file(audio_path, lang)
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
                    language=lang if lang not in {"auto", "detect", "none", None} else None,
                    language_confidence=None,
                    language_detected_by="live_fallback",
                )
            else:
                meeting.status = "failed"
                meeting.updated_at = datetime.now(timezone.utc)
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
                language=result.language
                if (result.language or "").strip().lower()
                not in {"", "auto", "detect", "none"}
                else None,
                language_confidence=result.language_confidence,
                language_detected_by="live_fallback",
            )

        asr.persist_transcript(db, meeting, result)
        meeting.updated_at = datetime.now(timezone.utc)
        db.commit()

        redis_store.clear_meeting_audio(meeting_id, keep_wav=True)
        audio.delete_raw_pcm(meeting_id)
        live_metrics.record_finalize(True)

        language_detection = None
        if result.language_detected_by or result.language_confidence is not None:
            language_detection = {
                "language": (result.language or meeting.language or None),
                "confidence": result.language_confidence,
                "detected_by": result.language_detected_by or "whisper",
            }
            if isinstance(language_detection["language"], str):
                detected = language_detection["language"].strip().lower()
                if detected in {"", "auto", "detect", "none"}:
                    language_detection["language"] = None

        return {
            "ok": True,
            "text": result.text,
            "segments": [
                {"text": s.text, "start": s.start, "end": s.end}
                for s in result.segments
            ],
            "engine": result.engine,
            "status": meeting.status,
            "language": meeting.language,
            "language_detection": language_detection,
        }
    except Exception as exc:
        logger.exception("Finalize failed for meeting %s", meeting_id)
        live_metrics.record_finalize(False)
        try:
            meeting = db.get(Meeting, meeting_id)
            if meeting and meeting.status == "processing":
                meeting.status = "failed"
                meeting.updated_at = datetime.now(timezone.utc)
                db.commit()
        except Exception:
            db.rollback()
        return {"ok": False, "message": str(exc)}
    finally:
        db.close()
