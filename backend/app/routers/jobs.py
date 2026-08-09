"""Background job status + enqueue endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..database import get_db
from ..deps import get_current_user
from ..models import User
from ..routers.meetings import _get_owned_meeting
from ..services import jobs
from ..services import finalize as finalize_svc
from ..services import asr
from ..services import transcription as transcription_svc

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


class EnqueueFinalizeRequest(BaseModel):
    meeting_id: str
    live_caption: str = ""


class EnqueueRetranscribeRequest(BaseModel):
    meeting_id: str


class JobStatusResponse(BaseModel):
    id: str
    type: str
    status: str
    created_at: float | None = None
    updated_at: float | None = None
    result: dict | None = None
    error: str | None = None
    dedupe_key: str = ""


def _register_handlers() -> None:
    def _finalize(payload: dict) -> dict:
        return finalize_svc.finalize_meeting_recording(
            payload.get("meeting_id") or "",
            payload.get("live_caption") or "",
            language=payload.get("language"),
        )

    def _retranscribe(payload: dict) -> dict:
        from ..database import SessionLocal
        from ..models import Meeting

        meeting_id = payload.get("meeting_id") or ""
        db = SessionLocal()
        try:
            meeting = db.get(Meeting, meeting_id)
            if meeting is None or not meeting.audio_path:
                raise RuntimeError("Meeting audio not available for retranscribe.")
            extra = transcription_svc.parse_custom_vocab(
                getattr(meeting, "custom_vocab", "") or ""
            )
            lang = transcription_svc.effective_asr_language(meeting.language)
            # Idempotent: persist_transcript deletes prior segment rows first.
            result = asr.transcribe_file(
                meeting.audio_path, lang, extra_terms=extra
            )
            asr.persist_transcript(db, meeting, result)
            db.commit()
            return {
                "ok": True,
                "text": result.text,
                "segments": [
                    {
                        "text": s.text,
                        "start": s.start,
                        "end": s.end,
                        "start_time": s.start,
                        "end_time": s.end,
                        "avg_logprob": s.avg_logprob,
                        "no_speech_prob": s.no_speech_prob,
                        "low_confidence": s.low_confidence,
                    }
                    for s in result.segments
                ],
            }
        finally:
            db.close()

    jobs.register("finalize", _finalize)
    jobs.register("retranscribe", _retranscribe)


_register_handlers()


@router.get("/{job_id}", response_model=JobStatusResponse)
def get_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
):
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    return JobStatusResponse(
        id=job["id"],
        type=job.get("type") or "",
        status=job.get("status") or "",
        created_at=job.get("created_at"),
        updated_at=job.get("updated_at"),
        result=job.get("result") if isinstance(job.get("result"), dict) else None,
        error=job.get("error"),
        dedupe_key=job.get("dedupe_key") or "",
    )


@router.post("/finalize", response_model=JobStatusResponse)
def enqueue_finalize(
    payload: EnqueueFinalizeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    meeting = _get_owned_meeting(payload.meeting_id, current_user, db)
    job = jobs.enqueue(
        "finalize",
        {
            "meeting_id": meeting.id,
            "live_caption": payload.live_caption,
            "language": meeting.language,
        },
        dedupe_key=f"finalize:{meeting.id}",
    )
    return JobStatusResponse(
        id=job["id"],
        type=job["type"],
        status=job["status"],
        created_at=job.get("created_at"),
        updated_at=job.get("updated_at"),
        dedupe_key=job.get("dedupe_key") or "",
    )


@router.post("/retranscribe", response_model=JobStatusResponse)
def enqueue_retranscribe(
    payload: EnqueueRetranscribeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    meeting = _get_owned_meeting(payload.meeting_id, current_user, db)
    job = jobs.enqueue(
        "retranscribe",
        {"meeting_id": meeting.id},
        dedupe_key=f"retranscribe:{meeting.id}",
    )
    return JobStatusResponse(
        id=job["id"],
        type=job["type"],
        status=job["status"],
        created_at=job.get("created_at"),
        updated_at=job.get("updated_at"),
        dedupe_key=job.get("dedupe_key") or "",
    )
