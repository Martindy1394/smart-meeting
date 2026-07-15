"""WebSocket endpoint for live PCM streaming + two-pass transcription.

Protocol (client -> server)
---------------------------
* Binary frames: raw little-endian 16-bit PCM @ 16 kHz mono.
* Text frames (JSON):
    {"type": "start"}                 begin/confirm a session
    {"type": "stop"}                  stop recording -> trigger finalization

Protocol (server -> client)
---------------------------
    {"type": "status", "transcription_available": bool, "message": str}
    {"type": "live_segment", "seq": int, "text": str, "start": float, "end": float}
    {"type": "live_caption", "text": str, "seq": int}   cumulative live caption
    {"type": "finalizing"}            finalization pass started
    {"type": "final_transcript", "text": str, "segments": [...]}
    {"type": "error", "message": str}

The connection is authenticated with a ``token`` query parameter (the JWT), and
the ``meeting_id`` query parameter selects the meeting to attach the transcript
to (must be owned by the authenticated user).
"""
from __future__ import annotations

import asyncio
import json
import logging

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..config import settings
from ..database import SessionLocal
from ..deps import get_user_from_token
from ..models import Meeting, TranscriptSegment
from ..services import audio, transcription

logger = logging.getLogger("smart_meeting.ws")

router = APIRouter()


async def _send(ws: WebSocket, payload: dict) -> None:
    await ws.send_text(json.dumps(payload))


def _finalize_blocking(meeting_id: str, pcm_bytes: bytes, language: str) -> dict:
    """Run the full-accuracy finalization pass and persist results (blocking)."""
    db = SessionLocal()
    try:
        meeting = db.get(Meeting, meeting_id)
        if meeting is None:
            return {"ok": False, "message": "Meeting disappeared during finalization."}

        # Persist the raw audio for archival / re-processing.
        audio_path = audio.save_wav(meeting_id, pcm_bytes)
        meeting.audio_path = audio_path
        meeting.duration_seconds = audio.wav_duration_seconds(pcm_bytes)
        meeting.status = "processing"
        db.commit()

        try:
            samples = audio.pcm16_to_float32(pcm_bytes)
            segments = transcription.transcribe_final(samples, language)
        except transcription.TranscriptionUnavailable as exc:
            meeting.status = "failed"
            db.commit()
            return {"ok": False, "message": str(exc)}

        full_text = " ".join(s.text for s in segments).strip()

        # Replace live segments with finalized, full-accuracy segments.
        db.query(TranscriptSegment).filter(
            TranscriptSegment.meeting_id == meeting_id
        ).delete()
        for i, seg in enumerate(segments):
            db.add(
                TranscriptSegment(
                    meeting_id=meeting_id,
                    kind="final",
                    text=seg.text,
                    start_time=seg.start,
                    end_time=seg.end,
                    seq=i,
                )
            )
        meeting.final_transcript = full_text
        meeting.status = "finalized"
        db.commit()
        return {
            "ok": True,
            "text": full_text,
            "segments": [
                {"text": s.text, "start": s.start, "end": s.end} for s in segments
            ],
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
) -> str:
    """Transcribe one live window and merge into the cumulative caption."""
    samples = audio.pcm16_to_float32(chunk)
    # Only skip completely empty / digital-silence frames. Quiet speech must
    # still reach Whisper so live captions are not artificially restricted.
    if samples.size == 0 or float(np.max(np.abs(samples))) < 0.0008:
        return live_caption

    segs = await asyncio.to_thread(transcription.transcribe_live, samples, language)
    window_text = " ".join(s.text for s in segs).strip()
    if not window_text:
        return live_caption

    merged = transcription.merge_live_caption(live_caption, window_text)
    if merged == live_caption:
        return live_caption

    await _send(
        websocket,
        {
            "type": "live_caption",
            "seq": seq,
            "text": merged,
        },
    )
    # Also keep legacy live_segment for older clients / persistence.
    await _send(
        websocket,
        {
            "type": "live_segment",
            "seq": seq,
            "text": window_text,
            "start": segs[0].start if segs else 0.0,
            "end": segs[-1].end if segs else 0.0,
        },
    )
    _persist_live_segment(meeting_id, seq, window_text)
    return merged


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

    live_available = transcription.is_available()
    await _send(
        websocket,
        {
            "type": "status",
            "transcription_available": live_available,
            "message": (
                "Live transcription active — continuous unrestricted captions."
                if live_available
                else "Whisper backend not installed — audio will still be saved "
                "and can be finalized once ML dependencies are available."
            ),
        },
    )

    all_pcm = bytearray()
    window = bytearray()
    seq = 0
    live_caption = ""

    window_seconds = max(2.0, float(settings.whisper_live_window_seconds))
    hop_seconds = max(0.5, float(settings.whisper_live_hop_seconds))
    if hop_seconds > window_seconds:
        hop_seconds = window_seconds

    bytes_per_sample_frame = 2  # int16 mono
    bytes_per_window = int(window_seconds * settings.audio_sample_rate * bytes_per_sample_frame)
    bytes_per_hop = int(hop_seconds * settings.audio_sample_rate * bytes_per_sample_frame)
    # Keep overlap so words spanning hop boundaries are not cut off.
    bytes_overlap = max(0, bytes_per_window - bytes_per_hop)

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            data = message.get("bytes")
            if data is not None:
                all_pcm.extend(data)
                window.extend(data)
                # Fire as soon as we have a full window; then advance by hop,
                # keeping overlap audio for the next pass.
                while live_available and len(window) >= bytes_per_window:
                    chunk = bytes(window[:bytes_per_window])
                    # Retain overlap for continuity; drop the consumed hop.
                    if bytes_overlap > 0:
                        window[:] = window[bytes_per_hop:]
                    else:
                        window.clear()
                    seq += 1
                    try:
                        live_caption = await _emit_live_window(
                            websocket,
                            meeting_id=meeting_id,
                            chunk=chunk,
                            language=language,
                            seq=seq,
                            live_caption=live_caption,
                        )
                    except Exception as exc:  # keep the stream alive on model errors
                        logger.exception("Live transcription error: %s", exc)
                continue

            text_data = message.get("text")
            if text_data:
                try:
                    control = json.loads(text_data)
                except json.JSONDecodeError:
                    continue
                if control.get("type") == "stop":
                    break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for meeting %s", meeting_id)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("WebSocket error: %s", exc)

    # Flush any leftover live audio before the final pass so trailing words
    # still appear in live captions (final pass will refine everything).
    if live_available and len(window) >= int(0.4 * settings.audio_sample_rate * 2):
        seq += 1
        try:
            live_caption = await _emit_live_window(
                websocket,
                meeting_id=meeting_id,
                chunk=bytes(window),
                language=language,
                seq=seq,
                live_caption=live_caption,
            )
        except Exception as exc:
            logger.exception("Live flush transcription error: %s", exc)

    # ---- Finalization (two-pass) -------------------------------------------
    try:
        await _send(websocket, {"type": "finalizing"})
    except Exception:
        pass

    if len(all_pcm) < 2:
        _mark_status(meeting_id, "finalized")
        try:
            await _send(
                websocket,
                {"type": "final_transcript", "text": "", "segments": []},
            )
            await websocket.close()
        except Exception:
            pass
        return

    result = await asyncio.to_thread(
        _finalize_blocking, meeting_id, bytes(all_pcm), language
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
                websocket, {"type": "error", "message": result.get("message", "Finalization failed.")}
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
