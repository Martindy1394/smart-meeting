"""Background cleanup for abandoned live recording sessions."""
from __future__ import annotations

import logging
import time

from ..config import settings
from ..database import SessionLocal
from ..models import Meeting
from . import audio, finalize, redis_store

logger = logging.getLogger("smart_meeting.janitor")


def sweep_abandoned_sessions() -> dict:
    """Finalize or clear sessions idle longer than ``abandoned_session_seconds``."""
    timeout = max(60, int(settings.abandoned_session_seconds))
    now = time.time()
    scanned = 0
    finalized = 0
    cleared = 0

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
        for meeting_id in meeting_ids:
            scanned += 1
            meeting = db.get(Meeting, meeting_id)
            if meeting is None:
                redis_store.clear_meeting_audio(meeting_id, keep_wav=False)
                audio.delete_raw_pcm(meeting_id)
                cleared += 1
                continue
            if meeting.status in ("finalized", "processing", "failed"):
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
                language=meeting.language,
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

    return {"scanned": scanned, "finalized": finalized, "cleared": cleared}


async def janitor_loop(stop_event) -> None:
    import asyncio

    interval = max(15.0, float(settings.session_janitor_interval_seconds))
    while not stop_event.is_set():
        try:
            stats = await asyncio.to_thread(sweep_abandoned_sessions)
            if stats.get("finalized") or stats.get("cleared"):
                logger.info("Janitor sweep %s", stats)
        except Exception:
            logger.exception("Janitor sweep failed")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            continue
