"""WebSocket endpoint for live PCM streaming + two-pass transcription.

Recorded audio is appended to Redis memory storage as it arrives. Live caption
state is also kept in Redis so reconnects can resume. On stop, PCM is read back
from Redis for the full-accuracy Whisper pass and WAV archival.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..config import settings
from ..database import SessionLocal
from ..deps import get_user_from_token
from ..models import Meeting, TranscriptSegment
from ..services import asr, audio, redis_store

logger = logging.getLogger("smart_meeting.ws")

router = APIRouter()


async def _send(ws: WebSocket, payload: dict) -> None:
    await ws.send_text(json.dumps(payload))


def _finalize_blocking(
    meeting_id: str,
    pcm_bytes: bytes,
    language: str,
    live_caption: str = "",
) -> dict:
    """Run the full-accuracy finalization pass and persist results (blocking)."""
    db = SessionLocal()
    try:
        meeting = db.get(Meeting, meeting_id)
        if meeting is None:
            return {"ok": False, "message": "Meeting disappeared during finalization."}

        # Persist the raw audio for archival / playback (disk + Redis WAV cache).
        audio_path = audio.save_wav(meeting_id, pcm_bytes)
        meeting.audio_path = audio_path
        meeting.duration_seconds = audio.wav_duration_seconds(pcm_bytes)
        meeting.status = "processing"
        db.commit()

        try:
            # Whisper ASR full-accuracy pass on the complete recording.
            result = asr.transcribe_pcm_bytes(pcm_bytes, language, live=False)
        except asr.ASRUnavailable as exc:
            # Fall back to the accumulated live caption rather than losing speech.
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
                    language=language,
                )
            else:
                meeting.status = "failed"
                db.commit()
                return {"ok": False, "message": str(exc)}

        # If the final pass is markedly shorter than live captions, keep the
        # richer live text so the finalized transcript is not incomplete.
        live = (live_caption or "").strip()
        final_text = (result.text or "").strip()
        live_words = len(live.split()) if live else 0
        final_words = len(final_text.split()) if final_text else 0
        if live_words > 0 and final_words < max(1, int(live_words * 0.6)):
            logger.warning(
                "Final ASR (%d words) shorter than live caption (%d words); "
                "preferring live caption for meeting %s",
                final_words,
                live_words,
                meeting_id,
            )
            result = asr.ASRResult(
                text=live,
                segments=[
                    asr.Segment(
                        text=live, start=0.0, end=meeting.duration_seconds or 0.0
                    )
                ],
                engine=f"{result.engine}+live-caption",
                language=language,
            )

        asr.persist_transcript(db, meeting, result)
        db.commit()

        # Live PCM buffer is no longer needed; keep the Redis WAV cache.
        redis_store.clear_meeting_audio(meeting_id, keep_wav=True)

        return {
            "ok": True,
            "text": result.text,
            "segments": [
                {"text": s.text, "start": s.start, "end": s.end}
                for s in result.segments
            ],
            "engine": result.engine,
        }
    finally:
        db.close()


async def _emit_live_window(
    websocket: WebSocket,
    *,
    meeting_id: str,
    chunk: bytes,
    language: str,
    seq: int,
    live_caption: str,
    previous_window: str = "",
) -> tuple[str, str]:
    """Transcribe one live window and merge into the cumulative caption.

    Returns ``(live_caption, window_text_used_as_previous)``.
    """
    samples = audio.pcm16_to_float32(chunk)
    # Only skip completely empty / digital-silence frames. Quiet speech must
    # still reach Whisper so live captions are not artificially restricted.
    if samples.size == 0 or float(np.max(np.abs(samples))) < 0.0008:
        return live_caption, previous_window

    result = await asyncio.to_thread(asr.transcribe_pcm, samples, language, live=True)
    window_text = result.text
    if not window_text:
        return live_caption, previous_window

    merged = asr.merge_live_caption(
        live_caption,
        window_text,
        previous_window=previous_window,
    )
    # Monotonic guard at the socket layer too.
    if len(merged.split()) < len((live_caption or "").split()):
        merged = live_caption
    if merged != live_caption:
        await _send(
            websocket,
            {
                "type": "live_caption",
                "seq": seq,
                "text": merged,
                "engine": "whisper",
            },
        )
        # Also keep legacy live_segment for older clients / persistence.
        segs = result.segments
        await _send(
            websocket,
            {
                "type": "live_segment",
                "seq": seq,
                "text": window_text,
                "start": segs[0].start if segs else 0.0,
                "end": segs[-1].end if segs else 0.0,
                "engine": "whisper",
            },
        )
        # Throttle SQLite writes during multi-hour board meetings.
        every = max(1, int(settings.live_segment_persist_every))
        if seq == 1 or seq % every == 0:
            _persist_live_segment(meeting_id, seq, window_text)
    return merged, window_text


@router.websocket("/ws/transcribe")
async def transcribe_ws(websocket: WebSocket):
    token = websocket.query_params.get("token")
    meeting_id = websocket.query_params.get("meeting_id")

    # Authenticate before accepting to avoid leaking an open socket.
    db = SessionLocal()
    try:
        user = get_user_from_token(token or "", db)
        meeting = db.get(Meeting, meeting_id) if meeting_id else None
        valid = (
            user is not None
            and meeting is not None
            and meeting.owner_id == user.id
        )
        language = meeting.language if meeting else settings.whisper_default_language
    finally:
        db.close()

    if not valid:
        await websocket.close(code=4401)
        return

    await websocket.accept()

    live_available = asr.is_available()
    redis_ok = redis_store.is_available()

    # Resume caption / offset state from Redis when reconnecting.
    meta = redis_store.get_session_meta(meeting_id) if redis_ok else {}
    live_caption = meta.get("live_caption") or ""
    previous_window = meta.get("previous_window") or ""
    seq = int(meta.get("seq") or 0)
    # Bytes already consumed by live windowing (exclusive end offset into Redis PCM).
    live_offset = int(meta.get("live_offset") or 0)

    # Local fallback only used when Redis is down.
    local_pcm = bytearray()
    window = bytearray()

    max_hours = float(settings.max_meeting_hours or 0)
    max_pcm_bytes = (
        int(max_hours * 3600 * settings.audio_sample_rate * 2) if max_hours > 0 else 0
    )

    await _send(
        websocket,
        {
            "type": "status",
            "transcription_available": live_available,
            "asr_engine": asr.engine_name(),
            "redis_audio": redis_ok,
            "live_caption": live_caption,
            "max_meeting_hours": max_hours or None,
            "keepalive_seconds": settings.ws_keepalive_seconds,
            "message": (
                "Whisper ASR active — Redis-buffered for multi-hour board meetings "
                f"(up to {max_hours:g}h); 10s live windows, language=tl / task=transcribe."
                if live_available and redis_ok
                else (
                    "Whisper ASR active — Redis unavailable, using in-process buffer "
                    "(prefer Redis for 4h+ meetings)."
                    if live_available
                    else "Whisper ASR not installed — audio will still be saved "
                    "and can be transcribed once ML dependencies are available."
                )
            ),
        },
    )
    if live_caption:
        await _send(
            websocket,
            {
                "type": "live_caption",
                "seq": seq,
                "text": live_caption,
                "engine": "whisper",
            },
        )

    window_seconds = max(2.0, float(settings.whisper_live_window_seconds))
    hop_seconds = max(0.5, float(settings.whisper_live_hop_seconds))
    if hop_seconds > window_seconds:
        hop_seconds = window_seconds

    bytes_per_sample_frame = 2  # int16 mono
    bytes_per_window = int(
        window_seconds * settings.audio_sample_rate * bytes_per_sample_frame
    )
    bytes_per_hop = int(
        hop_seconds * settings.audio_sample_rate * bytes_per_sample_frame
    )
    bytes_overlap = max(0, bytes_per_window - bytes_per_hop)
    warned_long = False
    recording_capped = False

    async def _process_live_from_redis() -> None:
        nonlocal live_caption, previous_window, seq, live_offset
        if not live_available:
            return
        total = redis_store.get_pcm_length(meeting_id)
        # Emit as soon as we have a full window ahead of live_offset.
        while total - live_offset >= bytes_per_window:
            chunk = redis_store.get_pcm_slice(
                meeting_id, live_offset, live_offset + bytes_per_window - 1
            )
            if len(chunk) < bytes_per_window:
                break
            seq += 1
            try:
                live_caption, previous_window = await _emit_live_window(
                    websocket,
                    meeting_id=meeting_id,
                    chunk=chunk,
                    language=language,
                    seq=seq,
                    live_caption=live_caption,
                    previous_window=previous_window,
                )
            except Exception as exc:
                logger.exception("Live transcription error: %s", exc)
            # Advance by hop; retain overlap in Redis by only moving the offset.
            live_offset += bytes_per_hop
            redis_store.set_session_meta(
                meeting_id,
                live_caption=live_caption,
                previous_window=previous_window,
                live_offset=live_offset,
                seq=seq,
            )
            total = redis_store.get_pcm_length(meeting_id)

    async def _keepalive_loop() -> None:
        interval = max(10.0, float(settings.ws_keepalive_seconds))
        try:
            while True:
                await asyncio.sleep(interval)
                await _send(
                    websocket,
                    {
                        "type": "ping",
                        "ts": time.time(),
                        "pcm_bytes": (
                            redis_store.get_pcm_length(meeting_id)
                            if redis_ok
                            else len(local_pcm)
                        ),
                    },
                )
        except Exception:
            return

    keepalive_task = asyncio.create_task(_keepalive_loop())

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            data = message.get("bytes")
            if data is not None:
                if recording_capped:
                    continue
                if redis_ok:
                    current_len = redis_store.get_pcm_length(meeting_id)
                    if max_pcm_bytes and current_len + len(data) > max_pcm_bytes:
                        recording_capped = True
                        hours = current_len / (
                            settings.audio_sample_rate * 2 * 3600
                        )
                        await _send(
                            websocket,
                            {
                                "type": "warning",
                                "message": (
                                    f"Reached the configured max meeting length "
                                    f"({max_hours:g}h, ~{hours:.1f}h recorded). "
                                    "Stop recording to finalize; raise "
                                    "MAX_MEETING_HOURS to extend."
                                ),
                            },
                        )
                        continue
                    # Primary path: every recorded chunk is saved to Redis.
                    await asyncio.to_thread(redis_store.append_pcm, meeting_id, data)
                    # Soft milestone warnings at 4h / 8h for board meetings.
                    new_len = redis_store.get_pcm_length(meeting_id)
                    hours = new_len / (settings.audio_sample_rate * 2 * 3600)
                    if not warned_long and hours >= 4.0:
                        warned_long = True
                        await _send(
                            websocket,
                            {
                                "type": "info",
                                "message": (
                                    f"Long meeting in progress (~{hours:.1f}h). "
                                    "Audio remains buffered in Redis; live captions continue."
                                ),
                            },
                        )
                    await _process_live_from_redis()
                else:
                    if max_pcm_bytes and len(local_pcm) + len(data) > max_pcm_bytes:
                        recording_capped = True
                        continue
                    local_pcm.extend(data)
                    window.extend(data)
                    while live_available and len(window) >= bytes_per_window:
                        chunk = bytes(window[:bytes_per_window])
                        if bytes_overlap > 0:
                            window[:] = window[bytes_per_hop:]
                        else:
                            window.clear()
                        seq += 1
                        try:
                            live_caption, previous_window = await _emit_live_window(
                                websocket,
                                meeting_id=meeting_id,
                                chunk=chunk,
                                language=language,
                                seq=seq,
                                live_caption=live_caption,
                                previous_window=previous_window,
                            )
                        except Exception as exc:
                            logger.exception("Live transcription error: %s", exc)
                continue

            text_data = message.get("text")
            if text_data:
                try:
                    control = json.loads(text_data)
                except json.JSONDecodeError:
                    continue
                ctype = control.get("type")
                if ctype in ("pong", "ping"):
                    # Client keepalive — ignore payload.
                    continue
                if ctype == "stop":
                    break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for meeting %s", meeting_id)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("WebSocket error: %s", exc)
    finally:
        keepalive_task.cancel()
        try:
            await keepalive_task
        except Exception:
            pass

    # Flush leftover live audio (from Redis offset or local window).
    if live_available:
        if redis_ok:
            total = redis_store.get_pcm_length(meeting_id)
            leftover = total - live_offset
            min_flush = int(0.4 * settings.audio_sample_rate * 2)
            if leftover >= min_flush:
                chunk = redis_store.get_pcm_slice(
                    meeting_id, live_offset, total - 1
                )
                seq += 1
                try:
                    live_caption, previous_window = await _emit_live_window(
                        websocket,
                        meeting_id=meeting_id,
                        chunk=chunk,
                        language=language,
                        seq=seq,
                        live_caption=live_caption,
                        previous_window=previous_window,
                    )
                    live_offset = total
                    redis_store.set_session_meta(
                        meeting_id,
                        live_caption=live_caption,
                        previous_window=previous_window,
                        live_offset=live_offset,
                        seq=seq,
                    )
                except Exception as exc:
                    logger.exception("Live flush transcription error: %s", exc)
        elif len(window) >= int(0.4 * settings.audio_sample_rate * 2):
            seq += 1
            try:
                live_caption, previous_window = await _emit_live_window(
                    websocket,
                    meeting_id=meeting_id,
                    chunk=bytes(window),
                    language=language,
                    seq=seq,
                    live_caption=live_caption,
                    previous_window=previous_window,
                )
            except Exception as exc:
                logger.exception("Live flush transcription error: %s", exc)

    # ---- Finalization (two-pass) -------------------------------------------
    try:
        await _send(websocket, {"type": "finalizing"})
    except Exception:
        pass

    if redis_ok:
        pcm_bytes = await asyncio.to_thread(redis_store.get_pcm, meeting_id)
    else:
        pcm_bytes = bytes(local_pcm)

    if len(pcm_bytes) < 2:
        fallback = (live_caption or "").strip()
        _mark_status(meeting_id, "finalized")
        try:
            await _send(
                websocket,
                {
                    "type": "final_transcript",
                    "text": fallback,
                    "segments": (
                        [{"text": fallback, "start": 0.0, "end": 0.0}]
                        if fallback
                        else []
                    ),
                },
            )
            await websocket.close()
        except Exception:
            pass
        return

    result = await asyncio.to_thread(
        _finalize_blocking,
        meeting_id,
        pcm_bytes,
        language,
        live_caption,
    )
    try:
        if result.get("ok"):
            await _send(
                websocket,
                {
                    "type": "final_transcript",
                    "text": result["text"],
                    "segments": result["segments"],
                },
            )
        else:
            await _send(
                websocket,
                {
                    "type": "error",
                    "message": result.get("message", "Finalization failed."),
                },
            )
        await websocket.close()
    except Exception:
        pass


def _persist_live_segment(meeting_id: str, seq: int, text: str) -> None:
    db = SessionLocal()
    try:
        db.add(
            TranscriptSegment(
                meeting_id=meeting_id, kind="live", text=text, seq=seq
            )
        )
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


def _mark_status(meeting_id: str, status_value: str) -> None:
    db = SessionLocal()
    try:
        meeting = db.get(Meeting, meeting_id)
        if meeting:
            meeting.status = status_value
            db.commit()
    finally:
        db.close()
