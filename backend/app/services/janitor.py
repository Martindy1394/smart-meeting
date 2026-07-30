"""Background cleanup for abandoned live recording sessions."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from ..config import settings
from ..database import SessionLocal
from ..models import Meeting
from . import audio, finalize, redis_store

logger = logging.getLogger("smart_meeting.janitor")


def sweep_abandoned_sessions() -> dict:
    """Finalize abandoned recordings and reclaim stuck ``processing`` meetings."""
    timeout = max(60, int(settings.abandoned_session_seconds))
    now = time.time()
    scanned = 0
    finalized = 0
    cleared = 0
    reclaimed = 0

    meeting_ids = set(redis_store.list_session_meeting_ids())
    # Also catch disk PCM left behind without Redis meta.
    try:
        import os

        audio_root = audio.audio_dir()
        for name in os.listdir(audio_root):
            if name.endswith(".pcm"):
                meeting_ids.add(name[:-4])
    except Exception:
        pass

    db = SessionLocal()
    try:
        # Reclaim Whisper jobs that never finished (process kill / OOM / crash).
        for meeting in (
            db.query(Meeting).filter(Meeting.status == "processing").all()
        ):
            if not finalize.is_processing_stale(meeting):
                continue
            has_text = bool((meeting.final_transcript or "").strip())
            meeting.status = "finalized" if has_text else "failed"
            meeting.updated_at = datetime.now(timezone.utc)
            logger.warning(
                "Janitor reclaiming stale processing meeting %s → %s",
                meeting.id,
                meeting.status,
            )
            reclaimed += 1
        if reclaimed:
            db.commit()

        for meeting_id in meeting_ids:
            scanned += 1
            meeting = db.get(Meeting, meeting_id)
            if meeting is None:
                redis_store.clear_meeting_audio(meeting_id, keep_wav=False)
                audio.delete_raw_pcm(meeting_id)
                cleared += 1
                continue
            if meeting.status == "processing":
                # Keep live buffers while ASR may still need PCM→WAV materialization.
                continue
            if meeting.status in ("finalized", "failed"):
                # Stale live buffers after a completed meeting.
                if audio.get_raw_pcm_length(meeting_id) or redis_store.get_pcm_length(
                    meeting_id
                ):
                    redis_store.clear_meeting_audio(meeting_id, keep_wav=True)
                    audio.delete_raw_pcm(meeting_id)
                    cleared += 1
                continue
            if meeting.status != "recording":
                continue

            meta = redis_store.get_session_meta(meeting_id)
            last_active = float(meta.get("last_active") or 0.0)
            if last_active <= 0:
                # Fall back to PCM mtime.
                path = audio.raw_pcm_path(meeting_id)
                try:
                    import os

                    last_active = os.path.getmtime(path) if os.path.exists(path) else 0.0
                except OSError:
                    last_active = 0.0
            if last_active <= 0 or (now - last_active) < timeout:
                continue

            logger.warning(
                "Janitor finalizing abandoned meeting %s (idle %.0fs)",
                meeting_id,
                now - last_active,
            )
            result = finalize.finalize_meeting_recording(
                meeting_id,
                meta.get("live_caption") or meeting.final_transcript or "",
                language=meeting.language or settings.whisper_default_language or "hil",
            )
            if result.get("ok"):
                finalized += 1
            else:
                logger.error(
                    "Janitor finalize failed for %s: %s",
                    meeting_id,
                    result.get("message"),
                )
    finally:
        db.close()

    return {
        "scanned": scanned,
        "finalized": finalized,
        "cleared": cleared,
        "reclaimed": reclaimed,
        "audio_purged": purge_expired_audio(),
    }


def purge_expired_audio() -> int:
    """Delete finalized meeting WAV/PCM older than ``audio_retention_days``.

    Transcripts, summaries, and translations are retained.
    """
    days = int(getattr(settings, "audio_retention_days", 0) or 0)
    if days <= 0:
        return 0
    cutoff = time.time() - (days * 86400)
    purged = 0
    db = SessionLocal()
    try:
        for meeting in (
            db.query(Meeting)
            .filter(Meeting.status.in_(("finalized", "failed")))
            .all()
        ):
            stamp = meeting.updated_at or meeting.created_at
            if stamp is None:
                continue
            try:
                ts = stamp.timestamp() if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc).timestamp()
            except Exception:
                continue
            if ts > cutoff:
                continue
            path = meeting.audio_path
            removed = False
            if path:
                try:
                    import os

                    abs_path = os.path.abspath(path)
                    if os.path.exists(abs_path):
                        os.remove(abs_path)
                        removed = True
                except OSError:
                    pass
                meeting.audio_path = None
            try:
                audio.delete_raw_pcm(meeting.id)
                redis_store.clear_meeting_audio(meeting.id, keep_wav=False)
                removed = True
            except Exception:
                pass
            if removed:
                purged += 1
        if purged:
            db.commit()
            logger.info("Audio retention purged %d meeting file(s) (>%d days)", purged, days)
    finally:
        db.close()
    return purged


async def janitor_loop(stop_event) -> None:
    import asyncio

    interval = max(15.0, float(settings.session_janitor_interval_seconds))
    while not stop_event.is_set():
        try:
            stats = await asyncio.to_thread(sweep_abandoned_sessions)
            if (
                stats.get("finalized")
                or stats.get("cleared")
                or stats.get("reclaimed")
            ):
                logger.info("Janitor sweep %s", stats)
        except Exception:
            logger.exception("Janitor sweep failed")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            continue
