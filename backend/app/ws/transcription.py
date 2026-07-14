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

# Process live captions in sequential windows (no overlap) so the UI can
# append each caption without repeating the same speech twice.
_LIVE_WINDOW_SECONDS = 2.5
_LIVE_HOP_SECONDS = 2.5
# Warn the client when the mic signal is effectively silent.
_SILENCE_RMS = 5e-4


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
            # Skip obviously empty captures early with a clear message.
            if audio.rms_level(samples) < 1e-4:
                meeting.status = "failed"
                db.commit()
                return {
                    "ok": False,
                    "message": (
                        "No usable microphone audio was captured (signal too quiet). "
                        "Check mic permissions/device and try again."
                    ),
                }
            segments = transcription.transcribe_final(samples, language)
        except transcription.TranscriptionUnavailable as exc:
            meeting.status = "failed"
            db.commit()
            return {"ok": False, "message": str(exc)}
        except Exception as exc:
            logger.exception("Final transcription failed: %s", exc)
            meeting.status = "failed"
            db.commit()
            return {"ok": False, "message": f"Final transcription failed: {exc}"}

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
                "Live transcription active."
                if live_available
                else "Whisper backend not installed — audio will still be saved "
                "and can be finalized once ML dependencies are available."
            ),
        },
    )

    all_pcm = bytearray()
    window = bytearray()
    seq = 0
    bytes_per_window = int(_LIVE_WINDOW_SECONDS * settings.audio_sample_rate * 2)
    bytes_per_hop = int(_LIVE_HOP_SECONDS * settings.audio_sample_rate * 2)
    silence_warned = False
    last_live_text = ""

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            data = message.get("bytes")
            if data is not None:
                all_pcm.extend(data)
                window.extend(data)
                if live_available and len(window) >= bytes_per_window:
                    chunk = bytes(window)
                    # Keep overlap for the next live pass.
                    keep = max(0, len(window) - bytes_per_hop)
                    window[:] = window[-keep:] if keep else b""
                    seq += 1
                    current_seq = seq
                    try:
                        samples = audio.pcm16_to_float32(chunk)
                        rms = audio.rms_level(samples)
                        if rms < _SILENCE_RMS:
                            if not silence_warned:
                                silence_warned = True
                                await _send(
                                    websocket,
                                    {
                                        "type": "warning",
                                        "message": (
                                            "Microphone signal is very quiet or silent. "
                                            "Check mic permissions, input device, and volume."
                                        ),
                                    },
                                )
                        else:
                            silence_warned = False
                            segs = await asyncio.to_thread(
                                transcription.transcribe_live, samples, language
                            )
                            text = " ".join(s.text for s in segs).strip()
                            # Skip duplicate captions from overlapping windows.
                            if text and text != last_live_text:
                                last_live_text = text
                                await _send(
                                    websocket,
                                    {
                                        "type": "live_segment",
                                        "seq": current_seq,
                                        "text": text,
                                        "start": segs[0].start if segs else 0.0,
                                        "end": segs[-1].end if segs else 0.0,
                                    },
                                )
                                _persist_live_segment(meeting_id, current_seq, text)
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
