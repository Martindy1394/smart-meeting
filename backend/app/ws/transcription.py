"""WebSocket endpoint for live PCM streaming + two-pass transcription.

Authoritative PCM is appended on disk. Redis keeps only a rolling recent window
plus session meta so reconnects can resume captions without Redis OOM on long
meetings. On explicit stop (WS or REST), disk PCM is finalized with Whisper.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..config import settings
from ..database import SessionLocal
from ..deps import get_user_from_token
from ..models import Meeting, TranscriptSegment
from ..services import asr, audio, finalize, live_metrics, redis_store

logger = logging.getLogger("smart_meeting.ws")

router = APIRouter()


async def _send(ws: WebSocket, payload: dict) -> None:
    await ws.send_text(json.dumps(payload))


async def _emit_live_window(
    websocket: WebSocket,
    *,
    meeting_id: str,
    chunk: bytes,
    language: str,
    seq: int,
    live_caption: str,
    previous_window: str = "",
    byte_offset: int | None = None,
    replace_caption: bool = False,
) -> tuple[str, str]:
    """Transcribe one live window and merge into the cumulative caption.

    Returns ``(live_caption, window_text_used_as_previous)``.

    ``replace_caption=True`` supersedes a short warmup caption with the first
    full window (avoids duplicated openings like ``Mic test! Mic test!``).
    """
    chunk = audio.align_pcm16(chunk)
    samples = audio.pcm16_to_float32(chunk)
    rms_i16 = audio.pcm_rms_int16(chunk)
    dur_s = len(chunk) / float(settings.audio_sample_rate * 2) if chunk else 0.0
    # Skip near-silent frames — softened so quiet board-mic speech is kept.
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if samples.size == 0 or peak < 0.004:
        logger.info(
            "live.window skip_silence meeting=%s seq=%d bytes=%d dur=%.2fs "
            "rms_i16=%.1f peak=%.4f offset=%s",
            meeting_id,
            seq,
            len(chunk),
            dur_s,
            rms_i16,
            peak,
            byte_offset,
        )
        # Clear stale previous_window after silence so the next hop merges cleanly.
        return live_caption, ""

    result = await asyncio.to_thread(asr.transcribe_pcm, samples, language, live=True)
    window_text = result.text
    logger.info(
        "live.window meeting=%s seq=%d bytes=%d dur=%.2fs rms_i16=%.1f "
        "offset=%s words=%d replace=%s text=%r",
        meeting_id,
        seq,
        len(chunk),
        dur_s,
        rms_i16,
        byte_offset,
        len((window_text or "").split()),
        replace_caption,
        (window_text or "")[:160],
    )
    if not window_text:
        return live_caption, previous_window

    if replace_caption:
        merged = window_text.strip()
    else:
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
        # Resolve auto → Hiligaynon by default so live ASR uses the PH prompt path
        # without requiring a manual Spoken language selection.
        from ..services import transcription as transcription_svc

        language = transcription_svc.effective_asr_language(
            meeting.language if meeting is not None else None
        )
        meeting_status = meeting.status if meeting else None
    finally:
        db.close()

    if not valid:
        await websocket.close(code=4401)
        return

    # Single-writer lock so two tabs cannot interleave PCM for the same meeting.
    lock_token = str(uuid.uuid4())
    if not redis_store.acquire_live_lock(meeting_id, lock_token):
        await websocket.accept()
        await _send(
            websocket,
            {
                "type": "error",
                "message": (
                    "This meeting is already being recorded in another tab "
                    "or session. Stop that recording first, or wait for the "
                    "lock to expire."
                ),
            },
        )
        await websocket.close(code=4409)
        return

    await websocket.accept()

    # Do not reopen a finished meeting — that used to look like "recording stopped"
    # after a long session when the client reconnected into a finalized meeting.
    if meeting_status in ("finalized", "processing", "failed"):
        await _send(
            websocket,
            {
                "type": "error",
                "message": (
                    f"Meeting is already {meeting_status}. "
                    "Start a new recording or re-transcribe from saved audio."
                ),
            },
        )
        redis_store.release_live_lock(meeting_id, lock_token)
        await websocket.close()
        return

    live_available = asr.is_available()
    redis_ok = redis_store.is_available()
    live_metrics.session_started()

    # Ensure the live model is loaded before the first window (non-blocking if warm).
    if live_available:
        from ..services import transcription as transcription_svc

        asyncio.create_task(asyncio.to_thread(transcription_svc.warm_live_model))

    # Resume caption / offset state from Redis when reconnecting.
    meta = redis_store.get_session_meta(meeting_id) if redis_ok else {}
    live_caption = meta.get("live_caption") or ""
    previous_window = meta.get("previous_window") or ""
    seq = int(meta.get("seq") or 0)
    # Bytes already consumed by live windowing (exclusive end offset into disk PCM).
    live_offset = int(meta.get("live_offset") or 0)
    warmup_done = bool(meta.get("seq"))  # skip warmup after reconnect mid-session
    resumed = bool(seq or live_caption or live_offset)

    # Local fallback only used when Redis is down (disk is still authoritative).
    local_pcm = bytearray()

    max_hours = float(settings.max_meeting_hours or 0)
    max_pcm_bytes = (
        int(max_hours * 3600 * settings.audio_sample_rate * 2) if max_hours > 0 else 0
    )

    if resumed:
        status_message = (
            f"Reconnected — resumed live captions (seq={seq}, "
            f"{len(live_caption.split()) if live_caption else 0} words buffered)."
        )
        logger.info(
            "WS resume meeting=%s seq=%d live_offset=%d caption_words=%d disk_pcm=%d",
            meeting_id,
            seq,
            live_offset,
            len(live_caption.split()) if live_caption else 0,
            audio.get_raw_pcm_length(meeting_id),
        )
    elif live_available and redis_ok:
        status_message = (
            "Whisper ASR active — disk PCM + Redis rolling buffer "
            f"(up to {max_hours:g}h); overlapping live windows, language={language}."
        )
    elif live_available:
        status_message = (
            "Whisper ASR active — Redis unavailable, using disk PCM "
            "(prefer Redis for reconnect resume)."
        )
    else:
        status_message = (
            "Whisper ASR not installed — audio will still be saved "
            "and can be transcribed once ML dependencies are available."
        )

    await _send(
        websocket,
        {
            "type": "status",
            "transcription_available": live_available,
            "asr_engine": asr.engine_name(),
            "redis_audio": redis_ok,
            "live_caption": live_caption,
            "resumed": resumed,
            "seq": seq,
            "live_offset": live_offset,
            "max_meeting_hours": max_hours or None,
            "keepalive_seconds": settings.ws_keepalive_seconds,
            "live_caption_prefer_ratio": settings.live_caption_prefer_ratio,
            "redis_live_buffer_seconds": settings.redis_live_buffer_seconds,
            "message": status_message,
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
    warmup_seconds = max(1.0, float(settings.whisper_live_warmup_seconds))
    bytes_per_warmup = int(
        warmup_seconds * settings.audio_sample_rate * bytes_per_sample_frame
    )
    # Never longer than the steady-state window.
    bytes_per_warmup = min(bytes_per_warmup, bytes_per_window)
    max_backlog = max(1, int(settings.live_asr_max_backlog_windows)) * bytes_per_hop
    warned_long = False
    warned_8h = False
    recording_capped = False
    explicit_stop = False
    redis_append_failures = 0

    def _recording_length() -> int:
        disk_len = audio.get_raw_pcm_length(meeting_id)
        if disk_len > 0:
            return disk_len
        if redis_ok:
            return redis_store.get_pcm_length(meeting_id)
        return len(local_pcm)

    def _read_slice(start: int, end_inclusive: int) -> bytes:
        piece = audio.get_raw_pcm_slice(meeting_id, start, end_inclusive)
        if piece:
            return piece
        if redis_ok:
            return redis_store.get_pcm_slice(meeting_id, start, end_inclusive)
        return bytes(local_pcm[start : end_inclusive + 1])

    def _persist_meta() -> None:
        if not redis_ok:
            return
        redis_store.set_session_meta(
            meeting_id,
            live_caption=live_caption,
            previous_window=previous_window,
            live_offset=live_offset,
            seq=seq,
        )

    def _apply_backpressure(total: int) -> None:
        nonlocal live_offset
        backlog = max(0, total - live_offset)
        live_metrics.set_backlog_windows(backlog // max(1, bytes_per_hop))
        if backlog <= max_backlog:
            return
        # Partial skip only — jumping all the way to the live edge was dropping
        # whole spoken phrases from live captions (disk/final still had them).
        skip_hops = max(1, int(settings.live_asr_backpressure_skip_hops))
        skip_bytes = skip_hops * bytes_per_hop
        target = live_offset + skip_bytes
        # Never skip past the last full window start.
        max_target = max(0, total - bytes_per_window)
        if bytes_per_hop > 0:
            max_target = (max_target // bytes_per_hop) * bytes_per_hop
        target = min(target, max_target)
        if target > live_offset:
            dropped = (target - live_offset) // max(1, bytes_per_hop)
            logger.warning(
                "Live ASR backpressure meeting=%s skip_hops=%d "
                "(backlog_bytes=%d max=%d new_offset=%d)",
                meeting_id,
                dropped,
                backlog,
                max_backlog,
                target,
            )
            live_metrics.record_windows_dropped(dropped)
            live_offset = target
            _persist_meta()

    async def _process_one_live_window() -> bool:
        """Transcribe at most one live window from disk PCM.

        Windows are ``[live_offset, live_offset+window)`` and the offset advances
        by ``hop`` only — the 5s overlap is retained in the buffer (never trimmed).
        """
        nonlocal live_caption, previous_window, seq, live_offset, warmup_done
        if not live_available:
            return False
        total = _recording_length()
        _apply_backpressure(total)

        # Fast first caption: short warmup before the first full 10s window.
        if (
            not warmup_done
            and live_offset == 0
            and total >= bytes_per_warmup
            and total < bytes_per_window
        ):
            chunk = _read_slice(0, bytes_per_warmup - 1)
            if len(chunk) >= bytes_per_warmup:
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
                        byte_offset=0,
                    )
                except Exception as exc:
                    logger.exception("Live warmup transcription error: %s", exc)
                warmup_done = True
                _persist_meta()
                return True
            return False

        if total - live_offset < bytes_per_window:
            return False

        chunk = _read_slice(live_offset, live_offset + bytes_per_window - 1)
        # Wait until the full window is on disk — never transcribe a partial hop.
        if len(chunk) < bytes_per_window:
            logger.debug(
                "live.window incomplete meeting=%s offset=%d need=%d got=%d",
                meeting_id,
                live_offset,
                bytes_per_window,
                len(chunk),
            )
            return False
        window_offset = live_offset
        # First full window after warmup re-covers t=0 — replace the short
        # warmup caption instead of appending a near-duplicate opening.
        supersede_warmup = bool(
            warmup_done and window_offset == 0 and (live_caption or "").strip()
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
                byte_offset=window_offset,
                replace_caption=supersede_warmup,
            )
        except Exception as exc:
            logger.exception("Live transcription error: %s", exc)
        warmup_done = True
        # Advance by hop only — overlap stays available for the next window.
        live_offset += bytes_per_hop
        _persist_meta()
        return True

    # Live ASR runs in a sibling task so Whisper never blocks keepalive pings.
    asr_wake = asyncio.Event()
    asr_stop = asyncio.Event()

    async def _live_asr_worker() -> None:
        if not live_available:
            return
        while not asr_stop.is_set():
            try:
                await asyncio.wait_for(asr_wake.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass
            asr_wake.clear()
            # Drain ready windows, but yield between each so the receive loop
            # can still ping / accept PCM. Bound drain so a huge backlog cannot
            # starve receive forever (backpressure also skips ahead).
            drained = 0
            while not asr_stop.is_set() and drained < max(
                1, int(settings.live_asr_max_backlog_windows)
            ):
                try:
                    did = await _process_one_live_window()
                except Exception as exc:
                    logger.exception("Live ASR worker error: %s", exc)
                    break
                if not did:
                    break
                drained += 1
                await asyncio.sleep(0)

    asr_task = asyncio.create_task(_live_asr_worker())

    keepalive_interval = max(10.0, float(settings.ws_keepalive_seconds))

    try:
        while True:
            try:
                message = await asyncio.wait_for(
                    websocket.receive(), timeout=keepalive_interval
                )
            except asyncio.TimeoutError:
                try:
                    await _send(
                        websocket,
                        {
                            "type": "ping",
                            "ts": time.time(),
                            "pcm_bytes": _recording_length(),
                        },
                    )
                except Exception:
                    break
                # Nudge ASR catch-up even when the client is quiet.
                asr_wake.set()
                continue

            if message.get("type") == "websocket.disconnect":
                break

            data = message.get("bytes")
            if data is not None:
                if recording_capped:
                    continue
                current_len = _recording_length()
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

                # Authoritative path: append every chunk to disk.
                total = await asyncio.to_thread(
                    audio.append_raw_pcm, meeting_id, data
                )

                if redis_ok:
                    # Rolling Redis window for reconnect assist (not full audio).
                    roll = await asyncio.to_thread(
                        redis_store.append_pcm, meeting_id, data
                    )
                    if roll < 0:
                        redis_append_failures += 1
                        if redis_append_failures in (1, 5, 20):
                            await _send(
                                websocket,
                                {
                                    "type": "warning",
                                    "message": (
                                        "Redis rolling-buffer append failed — "
                                        "disk recording continues; reconnect "
                                        "resume may be limited."
                                    ),
                                },
                            )
                    else:
                        redis_append_failures = 0
                    redis_store.touch_session(meeting_id)
                    redis_store.refresh_live_lock(meeting_id, lock_token)
                else:
                    # Disk is authoritative; keep a short local length mirror only.
                    local_pcm.extend(data)

                # Soft milestone warnings for long board meetings.
                hours = total / (settings.audio_sample_rate * 2 * 3600)
                if not warned_long and hours >= 4.0:
                    warned_long = True
                    await _send(
                        websocket,
                        {
                            "type": "info",
                            "message": (
                                f"Long meeting in progress (~{hours:.1f}h). "
                                "Audio is on disk; live captions continue."
                            ),
                        },
                    )
                if not warned_8h and hours >= 8.0:
                    warned_8h = True
                    await _send(
                        websocket,
                        {
                            "type": "info",
                            "message": (
                                f"Meeting past 8 hours (~{hours:.1f}h). "
                                "Recording is still active on disk."
                            ),
                        },
                    )
                # Disk-backed ASR worker is the single consumer (Redis up or down).
                asr_wake.set()
                continue

            text_data = message.get("text")
            if text_data:
                try:
                    control = json.loads(text_data)
                except json.JSONDecodeError:
                    continue
                ctype = control.get("type")
                if ctype in ("pong", "ping", "start", "pause", "resume"):
                    # Client keepalive / session hello / pause-play — ignore payload.
                    if redis_ok:
                        redis_store.touch_session(meeting_id)
                    continue
                if ctype == "stop":
                    explicit_stop = True
                    break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for meeting %s", meeting_id)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("WebSocket error: %s", exc)
    finally:
        asr_stop.set()
        asr_wake.set()
        try:
            await asyncio.wait_for(asr_task, timeout=120.0)
        except (Exception, asyncio.CancelledError):
            asr_task.cancel()
            try:
                await asr_task
            except (Exception, asyncio.CancelledError):
                pass
        _persist_meta()
        live_metrics.session_ended()
        redis_store.release_live_lock(meeting_id, lock_token)

    # Unexpected drops must NOT finalize — REST /stop or reconnect handles it.
    if not explicit_stop:
        logger.info(
            "WS closed without stop for meeting %s — keeping disk PCM "
            "(pcm=%s bytes, caption_words=%d) for reconnect or REST /stop",
            meeting_id,
            _recording_length(),
            len((live_caption or "").split()),
        )
        return

    # Flush leftover live audio in overlapping windows (never one giant chunk —
    # Whisper live models truncate / drop speech on multi-minute buffers).
    if live_available:
        total = _recording_length()
        min_flush = int(0.4 * settings.audio_sample_rate * 2)
        flush_guard = 0
        while total - live_offset >= bytes_per_window and flush_guard < 500:
            chunk = _read_slice(live_offset, live_offset + bytes_per_window - 1)
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
                    byte_offset=live_offset,
                )
            except Exception as exc:
                logger.exception("Live flush transcription error: %s", exc)
            live_offset += bytes_per_hop
            flush_guard += 1
            _persist_meta()
        leftover = total - live_offset
        if leftover >= min_flush:
            chunk = _read_slice(live_offset, total - 1)
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
                    byte_offset=live_offset,
                )
                live_offset = total
                _persist_meta()
            except Exception as exc:
                logger.exception("Live flush transcription error: %s", exc)

    # ---- Finalization (two-pass) — only after explicit client stop ----------
    try:
        await _send(websocket, {"type": "finalizing"})
    except Exception:
        pass

    result = await asyncio.to_thread(
        finalize.finalize_meeting_recording,
        meeting_id,
        live_caption,
        language=language,
    )
    try:
        if result.get("ok"):
            await _send(
                websocket,
                {
                    "type": "final_transcript",
                    "text": result.get("text") or "",
                    "segments": result.get("segments") or [],
                    "language": result.get("language"),
                    "language_detection": result.get("language_detection"),
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
