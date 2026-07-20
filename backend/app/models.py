"""SQLAlchemy ORM models: User, Meeting, TranscriptSegment."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    # Login credential (unique). Working email is separate contact info.
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    first_name: Mapped[str] = mapped_column(String(128), default="")
    last_name: Mapped[str] = mapped_column(String(128), default="")
    position: Mapped[str] = mapped_column(String(128), default="")
    workplace: Mapped[str] = mapped_column(String(255), default="")
    # Kept for older clients / display convenience.
    full_name: Mapped[str] = mapped_column(String(255), default="")
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    meetings: Mapped[list["Meeting"]] = relationship(
        back_populates="owner", cascade="all, delete-orphan"
    )

    def display_name(self) -> str:
        name = f"{self.first_name} {self.last_name}".strip()
        return name or self.full_name or self.username


class Meeting(Base):
    __tablename__ = "meetings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    owner_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    title: Mapped[str] = mapped_column(String(255), default="Untitled meeting")

    # Meeting details.
    venue: Mapped[str] = mapped_column(String(255), default="")
    # Scheduled date/time of the meeting (distinct from created_at).
    meeting_date: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # JSON-encoded list of attendee names.
    attendees: Mapped[str] = mapped_column(Text, default="[]")

    # Transcript status: recording | processing | finalized | failed
    status: Mapped[str] = mapped_column(String(32), default="recording")
    language: Mapped[str] = mapped_column(String(16), default="auto")
    # Whisper language-detect metadata (confidence 0–1; detected_by source).
    language_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    language_detected_by: Mapped[str] = mapped_column(String(32), default="")

    # Full-accuracy finalized transcript (post two-pass finalization).
    final_transcript: Mapped[str] = mapped_column(Text, default="")

    # Cached AI outputs.
    summary: Mapped[str] = mapped_column(Text, default="")
    summary_format: Mapped[str] = mapped_column(String(32), default="")
    translation: Mapped[str] = mapped_column(Text, default="")
    translation_language: Mapped[str] = mapped_column(String(64), default="")

    audio_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    owner: Mapped["User"] = relationship(back_populates="meetings")
    segments: Mapped[list["TranscriptSegment"]] = relationship(
        back_populates="meeting",
        cascade="all, delete-orphan",
        order_by="TranscriptSegment.start_time",
    )


class TranscriptSegment(Base):
    """A single decoded transcript fragment.

    ``kind`` distinguishes low-latency live captions ("live") from the
    full-accuracy finalized segments ("final") produced by the second pass.
    """

    __tablename__ = "transcript_segments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    meeting_id: Mapped[str] = mapped_column(
        ForeignKey("meetings.id", ondelete="CASCADE"), index=True, nullable=False
    )
    kind: Mapped[str] = mapped_column(String(16), default="live")
    text: Mapped[str] = mapped_column(Text, default="")
    start_time: Mapped[float] = mapped_column(Float, default=0.0)
    end_time: Mapped[float] = mapped_column(Float, default=0.0)
    seq: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    meeting: Mapped["Meeting"] = relationship(back_populates="segments")
