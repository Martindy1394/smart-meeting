"""Meeting management: list, create, read, update, delete, search."""
from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import or_
from sqlalchemy.orm import Session

from ..database import get_db
from ..deps import get_current_user
from ..models import Meeting, User
from ..schemas import (
    MeetingCreate,
    MeetingDetail,
    MeetingSummary,
    MeetingUpdate,
)

router = APIRouter(prefix="/api/meetings", tags=["meetings"])


def _get_owned_meeting(meeting_id: str, user: User, db: Session) -> Meeting:
    meeting = db.get(Meeting, meeting_id)
    if meeting is None or meeting.owner_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Meeting not found."
        )
    return meeting


def _to_summary(m: Meeting) -> MeetingSummary:
    return MeetingSummary(
        id=m.id,
        title=m.title,
        status=m.status,
        language=m.language,
        duration_seconds=m.duration_seconds,
        created_at=m.created_at,
        updated_at=m.updated_at,
        has_summary=bool(m.summary),
        has_translation=bool(m.translation),
    )


@router.get("", response_model=list[MeetingSummary])
def list_meetings(
    search: str | None = Query(default=None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(Meeting).filter(Meeting.owner_id == current_user.id)
    if search:
        pattern = f"%{search.strip()}%"
        query = query.filter(or_(Meeting.title.ilike(pattern)))
    meetings = query.order_by(Meeting.created_at.desc()).all()
    return [_to_summary(m) for m in meetings]


@router.post("", response_model=MeetingDetail, status_code=status.HTTP_201_CREATED)
def create_meeting(
    payload: MeetingCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    meeting = Meeting(
        owner_id=current_user.id,
        title=payload.title.strip() or "Untitled meeting",
        language=payload.language or "hil",
        status="recording",
    )
    db.add(meeting)
    db.commit()
    db.refresh(meeting)
    return MeetingDetail.model_validate(meeting)


@router.get("/{meeting_id}", response_model=MeetingDetail)
def get_meeting(
    meeting_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    meeting = _get_owned_meeting(meeting_id, current_user, db)
    return MeetingDetail.model_validate(meeting)


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
    db.commit()
    db.refresh(meeting)
    return MeetingDetail.model_validate(meeting)


@router.delete("/{meeting_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_meeting(
    meeting_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    meeting = _get_owned_meeting(meeting_id, current_user, db)
    # Best-effort removal of the stored audio file.
    if meeting.audio_path and os.path.exists(meeting.audio_path):
        try:
            os.remove(meeting.audio_path)
        except OSError:
            pass
    db.delete(meeting)
    db.commit()
    return None
