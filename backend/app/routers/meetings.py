"""Meeting management: list, create, read, update, delete, search, audio, ASR."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import or_
from sqlalchemy.orm import Session

from ..config import settings
from ..database import SessionLocal, get_db
from ..deps import get_current_user
from ..models import Meeting, User
from ..schemas import (
    MeetingCreate,
    MeetingDetail,
    MeetingSummary,
    MeetingUpdate,
)
from ..services import asr, audio, finalize, redis_store

logger = logging.getLogger("smart_meeting.meetings")

router = APIRouter(prefix="/api/meetings", tags=["meetings"])

# Cap uploads so a single request cannot fill the disk (≈ 30 min of 16 kHz mono WAV).
_MAX_UPLOAD_BYTES = 60 * 1024 * 1024
_UPLOAD_READ_CHUNK = 1024 * 1024


def _get_owned_meeting(meeting_id: str, user: User, db: Session) -> Meeting:
    meeting = db.get(Meeting, meeting_id)
    if meeting is None or meeting.owner_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Meeting not found."
        )
    return meeting


def _has_audio(m: Meeting) -> bool:
    if m.audio_path and os.path.exists(os.path.abspath(m.audio_path)):
        return True
    # Live recording PCM on disk, Redis rolling buffer, or cached WAV.
    return bool(
        audio.get_raw_pcm_length(m.id) > 0
        or redis_store.get_wav_bytes(m.id)
        or redis_store.get_pcm_length(m.id) > 0
    )


def _ensure_meeting_wav(meeting: Meeting, db: Session) -> str:
    """Return an on-disk WAV path, materializing from PCM/Redis when needed."""
    if meeting.audio_path:
        path = os.path.abspath(meeting.audio_path)
        if os.path.exists(path):
            return path

    pcm = audio.read_raw_pcm(meeting.id)
    if not pcm:
        pcm = redis_store.get_pcm(meeting.id)
    if pcm and len(pcm) >= 2:
        path = audio.save_wav(meeting.id, pcm)
        meeting.audio_path = path
        meeting.duration_seconds = audio.wav_duration_seconds(pcm)
        db.commit()
        return path

    wav_bytes = redis_store.get_wav_bytes(meeting.id)
    if wav_bytes:
        path = os.path.join(audio.audio_dir(), f"{meeting.id}.wav")
        with open(path, "wb") as fh:
            fh.write(wav_bytes)
        path = os.path.abspath(path)
        meeting.audio_path = path
        db.commit()
        return path

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="No audio recording is available to transcribe.",
    )


def _prepare_whisper_job(meeting: Meeting, db: Session) -> tuple[str, str]:
    """Validate meeting audio and mark processing. Returns ``(path, language)``."""
    if not asr.is_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Whisper ASR is not available on this server.",
        )
    # Resolve path *before* flipping status so a crash cannot leave a stuck
    # processing row with no audio_path.
    path = _ensure_meeting_wav(meeting, db)
    meeting.status = "processing"
    meeting.updated_at = datetime.now(timezone.utc)
    db.commit()
    # Honor meeting language (hil/tl). ``auto`` resolves to Hiligaynon by default
    # via transcription.effective_asr_language — no manual Spoken language needed.
    from ..services import transcription as transcription_svc

    lang = transcription_svc.effective_asr_language(meeting.language)
    return path, lang


async def _read_upload_capped(file: UploadFile) -> bytes:
    """Read an upload while enforcing the size cap before the full body lands."""
    content_length = getattr(file, "size", None)
    if content_length is None:
        header = file.headers.get("content-length") if file.headers else None
        if header:
            try:
                content_length = int(header)
            except ValueError:
                content_length = None
    if content_length is not None and content_length > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Audio file is too large (max 60 MB).",
        )

    chunks: list[bytes] = []
    total = 0
    while True:
        piece = await file.read(_UPLOAD_READ_CHUNK)
        if not piece:
            break
        total += len(piece)
        if total > _MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Audio file is too large (max 60 MB).",
            )
        chunks.append(piece)
    return b"".join(chunks)


def _clean_attendees(names: list[str]) -> str:
    cleaned = [n.strip() for n in names if isinstance(n, str) and n.strip()]
    # De-duplicate while preserving order.
    seen: set[str] = set()
    unique = [n for n in cleaned if not (n in seen or seen.add(n))]
    return json.dumps(unique)


def _language_detection_info(m: Meeting):
    """Build the nested language_detection payload from ORM columns."""
    from ..schemas import LanguageDetectionInfo

    by = (getattr(m, "language_detected_by", None) or "").strip()
    conf = getattr(m, "language_confidence", None)
    lang = (m.language or "").strip().lower()
    if not by and conf is None:
        return None
    return LanguageDetectionInfo(
        language=lang if lang and lang not in {"auto", "detect", "none"} else None,
        confidence=conf,
        detected_by=by or "whisper",
    )


def _to_summary(m: Meeting) -> MeetingSummary:
    summary_text = (m.summary or "").strip()
    translation_text = (m.translation or "").strip()
    return MeetingSummary(
        id=m.id,
        title=m.title,
        status=m.status,
        language=m.language,
        language_detection=_language_detection_info(m),
        venue=m.venue or "",
        meeting_date=m.meeting_date,
        duration_seconds=m.duration_seconds,
        created_at=m.created_at,
        updated_at=m.updated_at,
        summary=summary_text,
        summary_format=(m.summary_format or "").strip(),
        translation=translation_text,
        translation_language=(m.translation_language or "").strip(),
        has_summary=bool(summary_text),
        has_translation=bool(translation_text),
        has_audio=_has_audio(m),
        has_transcript=bool((m.final_transcript or "").strip()),
    )


def _to_detail(m: Meeting) -> MeetingDetail:
    detail = MeetingDetail.model_validate(m)
    detail.has_audio = _has_audio(m)
    detail.language_detection = _language_detection_info(m)
    return detail


def _persist_whisper_result(
    meeting: Meeting, db: Session, result: asr.ASRResult
) -> MeetingDetail:
    asr.persist_transcript(db, meeting, result)
    meeting.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(meeting)
    return _to_detail(meeting)


def _fail_whisper(meeting: Meeting, db: Session) -> None:
    meeting.status = "failed"
    meeting.updated_at = datetime.now(timezone.utc)
    db.commit()


def _run_whisper_on_meeting(meeting: Meeting, db: Session) -> MeetingDetail:
    """Full-accuracy Whisper ASR on the meeting's saved audio file (sync)."""
    path, language = _prepare_whisper_job(meeting, db)
    try:
        result = asr.transcribe_file(path, language)
    except asr.ASRUnavailable as exc:
        _fail_whisper(meeting, db)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        )
    except audio.AudioFormatError as exc:
        _fail_whisper(meeting, db)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        )
    except Exception as exc:
        logger.exception("Whisper ASR failed for meeting %s", meeting.id)
        _fail_whisper(meeting, db)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Whisper ASR failed: {exc}",
        )
    return _persist_whisper_result(meeting, db, result)


async def _run_whisper_on_meeting_async(
    meeting: Meeting, db: Session
) -> MeetingDetail:
    """Async variant: run blocking ASR in a worker thread, keep DB on this task."""
    path, language = _prepare_whisper_job(meeting, db)
    try:
        result = await asyncio.to_thread(asr.transcribe_file, path, language)
    except asr.ASRUnavailable as exc:
        _fail_whisper(meeting, db)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        )
    except audio.AudioFormatError as exc:
        _fail_whisper(meeting, db)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        )
    except Exception as exc:
        logger.exception("Whisper ASR failed for meeting %s", meeting.id)
        _fail_whisper(meeting, db)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Whisper ASR failed: {exc}",
        )
    return _persist_whisper_result(meeting, db, result)


async def _background_retranscribe(meeting_id: str, path: str, language: str) -> None:
    """Run Whisper off the request so long CPU jobs do not drop the HTTP client."""
    db = SessionLocal()
    try:
        meeting = db.get(Meeting, meeting_id)
        if meeting is None:
            return
        try:
            result = await asyncio.to_thread(asr.transcribe_file, path, language)
            _persist_whisper_result(meeting, db, result)
            logger.info("Background retranscribe finished meeting=%s", meeting_id)
        except Exception:
            logger.exception("Background retranscribe failed meeting=%s", meeting_id)
            meeting = db.get(Meeting, meeting_id)
            if meeting is not None:
                _fail_whisper(meeting, db)
    finally:
        db.close()


@router.get("", response_model=list[MeetingSummary])
def list_meetings(
    search: str | None = Query(default=None),
    has_audio: bool | None = Query(default=None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(Meeting).filter(Meeting.owner_id == current_user.id)
    if search:
        pattern = f"%{search.strip()}%"
        query = query.filter(or_(Meeting.title.ilike(pattern)))
    meetings = query.order_by(Meeting.created_at.desc()).all()
    summaries = [_to_summary(m) for m in meetings]
    if has_audio is True:
        summaries = [s for s in summaries if s.has_audio]
    elif has_audio is False:
        summaries = [s for s in summaries if not s.has_audio]
    return summaries


@router.post("", response_model=MeetingDetail, status_code=status.HTTP_201_CREATED)
def create_meeting(
    payload: MeetingCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    meeting = Meeting(
        owner_id=current_user.id,
        title=payload.title.strip(),
        language=(payload.language or "auto").strip().lower() or "auto",
        venue=payload.venue.strip(),
        meeting_date=payload.meeting_date or datetime.now(timezone.utc),
        attendees=_clean_attendees(payload.attendees),
        status="recording",
    )
    db.add(meeting)
    db.commit()
    db.refresh(meeting)
    return _to_detail(meeting)


@router.get("/{meeting_id}", response_model=MeetingDetail)
def get_meeting(
    meeting_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    meeting = _get_owned_meeting(meeting_id, current_user, db)
    return _to_detail(meeting)


@router.get("/{meeting_id}/audio")
def get_meeting_audio(
    meeting_id: str,
    download: bool = Query(default=False),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Stream the saved WAV recording for playback or download.

    Prefers disk archive; falls back to materializing WAV from PCM/Redis, then
    streams from disk so multi-hour recordings are not loaded into RAM.
    """
    meeting = _get_owned_meeting(meeting_id, current_user, db)
    safe_title = (meeting.title or "meeting").strip() or "meeting"
    safe_title = "".join(
        ch if ch.isalnum() or ch in " -_" else "_" for ch in safe_title
    )
    filename = f"{safe_title}.wav"
    disposition = "attachment" if download else "inline"

    try:
        path = _ensure_meeting_wav(meeting, db)
    except HTTPException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No audio recording is available for this meeting yet.",
        ) from None

    return FileResponse(
        path=path,
        media_type="audio/wav",
        filename=filename,
        content_disposition_type=disposition,
    )


@router.post("/{meeting_id}/audio", response_model=MeetingDetail)
async def upload_meeting_audio(
    meeting_id: str,
    file: UploadFile = File(...),
    transcribe: bool = Query(
        default=True,
        description="When true, run Whisper ASR on the uploaded audio immediately.",
    ),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Upload a WAV/PCM recording and optionally start Whisper ASR on it."""
    meeting = _get_owned_meeting(meeting_id, current_user, db)
    data = await _read_upload_capped(file)
    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded audio file is empty.",
        )

    try:
        # ffmpeg / WAV decode is blocking — keep the event loop free.
        path, duration = await asyncio.to_thread(
            audio.save_uploaded_audio,
            meeting_id,
            data,
            file.filename or "",
        )
    except audio.AudioFormatError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    meeting.audio_path = path
    meeting.duration_seconds = duration
    # Clear previous transcript until Whisper re-processes (if requested).
    if transcribe:
        meeting.final_transcript = ""
        meeting.status = "processing"
        meeting.updated_at = datetime.now(timezone.utc)
    db.commit()

    if not transcribe:
        db.refresh(meeting)
        return _to_detail(meeting)

    # Same pattern as retranscribe — return immediately and poll for status.
    asyncio.create_task(_background_retranscribe(meeting.id, path, "auto"))
    db.refresh(meeting)
    return _to_detail(meeting)


@router.post("/{meeting_id}/retranscribe", response_model=MeetingDetail)
async def retranscribe_meeting(
    meeting_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Start Whisper ASR on the saved WAV and return immediately.

    Full-accuracy ASR can take minutes on CPU. The meeting stays
    ``processing`` until the background job finishes; clients should poll
    ``GET /meetings/{id}`` rather than waiting on this request.
    """
    meeting = _get_owned_meeting(meeting_id, current_user, db)
    if meeting.status == "processing" and not finalize.is_processing_stale(meeting):
        # Idempotent — a job is already running.
        return _to_detail(meeting)
    path, language = _prepare_whisper_job(meeting, db)
    asyncio.create_task(_background_retranscribe(meeting.id, path, language))
    db.refresh(meeting)
    return _to_detail(meeting)


@router.post("/{meeting_id}/stop")
async def stop_meeting_recording(
    meeting_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Finalize a live recording without relying on the WebSocket stop message.

    Use this when the socket dropped before ``{type: stop}`` could be sent
    (tab close, reconnect gap, proxy drop). Idempotent if already finalized.
    """
    from ..services import finalize

    meeting = _get_owned_meeting(meeting_id, current_user, db)
    if meeting.status in ("finalized",):
        return {
            "ok": True,
            "already_done": True,
            "text": meeting.final_transcript or "",
            "status": meeting.status,
        }
    if meeting.status == "processing" and not finalize.is_processing_stale(meeting):
        return {
            "ok": True,
            "already_done": True,
            "in_progress": True,
            "text": meeting.final_transcript or "",
            "status": meeting.status,
            "message": "Transcription is already in progress.",
        }

    meta = redis_store.get_session_meta(meeting_id)
    live_caption = meta.get("live_caption") or ""
    result = await asyncio.to_thread(
        finalize.finalize_meeting_recording,
        meeting_id,
        live_caption,
        language=meeting.language or settings.whisper_default_language or "hil",
    )
    if not result.get("ok"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get("message") or "Finalization failed.",
        )
    db.refresh(meeting)
    return {
        "ok": True,
        "already_done": bool(result.get("already_done")),
        "in_progress": bool(result.get("in_progress")),
        "text": result.get("text") or meeting.final_transcript or "",
        "segments": result.get("segments") or [],
        "engine": result.get("engine"),
        "status": meeting.status,
        "meeting": _to_detail(meeting),
    }


@router.patch("/{meeting_id}", response_model=MeetingDetail)
def update_meeting(
    meeting_id: str,
    payload: MeetingUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    meeting = _get_owned_meeting(meeting_id, current_user, db)
    if payload.title is not None:
        meeting.title = payload.title.strip() or meeting.title
    if payload.venue is not None:
        meeting.venue = payload.venue.strip()
    if payload.meeting_date is not None:
        meeting.meeting_date = payload.meeting_date
    if payload.attendees is not None:
        meeting.attendees = _clean_attendees(payload.attendees)
    if payload.language is not None:
        lang = (payload.language or "").strip().lower()
        if lang in {
            "auto",
            "detect",
            "hil",
            "hiligaynon",
            "ilonggo",
            "tl",
            "tagalog",
            "fil",
            "filipino",
            "en",
            "english",
        }:
            # Normalize aliases to stable meeting labels.
            if lang in {"detect"}:
                lang = "auto"
            elif lang in {"hiligaynon", "ilonggo"}:
                lang = "hil"
            elif lang in {"tagalog", "fil", "filipino"}:
                lang = "tl"
            elif lang == "english":
                lang = "en"
            meeting.language = lang
    db.commit()
    db.refresh(meeting)
    return _to_detail(meeting)


@router.delete("/{meeting_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_meeting(
    meeting_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    meeting = _get_owned_meeting(meeting_id, current_user, db)
    # Best-effort removal of the stored audio file + Redis memory buffers.
    if meeting.audio_path and os.path.exists(meeting.audio_path):
        try:
            os.remove(meeting.audio_path)
        except OSError:
            pass
    redis_store.clear_meeting_audio(meeting_id, keep_wav=False)
    audio.delete_raw_pcm(meeting_id)
    db.delete(meeting)
    db.commit()
    return None
