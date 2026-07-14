"""Pydantic request/response schemas + validation rules."""
from __future__ import annotations

import re
from datetime import datetime

from pydantic import BaseModel, EmailStr, Field, field_validator

# At least 8 chars, one number and one special character (per requirements).
_PASSWORD_RE = re.compile(r"^(?=.*\d)(?=.*[^A-Za-z0-9]).{8,}$")


# ----------------------------- Auth ----------------------------------------
class SignupRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: str = Field(default="", max_length=255)

    @field_validator("password")
    @classmethod
    def _validate_password(cls, v: str) -> str:
        if not _PASSWORD_RE.match(v):
            raise ValueError(
                "Password must be at least 8 characters and include at least "
                "one number and one special character."
            )
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: "UserResponse"


class UserResponse(BaseModel):
    id: str
    email: EmailStr
    full_name: str
    created_at: datetime

    class Config:
        from_attributes = True


# --------------------------- Meetings --------------------------------------
class MeetingCreate(BaseModel):
    title: str = Field(default="Untitled meeting", max_length=255)
    language: str = Field(default="hil", max_length=16)


class MeetingUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=255)


class TranscriptSegmentResponse(BaseModel):
    id: str
    kind: str
    text: str
    start_time: float
    end_time: float
    seq: int

    class Config:
        from_attributes = True


class MeetingSummary(BaseModel):
    """Lightweight meeting representation for list views."""

    id: str
    title: str
    status: str
    language: str
    duration_seconds: float
    created_at: datetime
    updated_at: datetime
    has_summary: bool = False
    has_translation: bool = False

    class Config:
        from_attributes = True


class MeetingDetail(BaseModel):
    id: str
    title: str
    status: str
    language: str
    final_transcript: str
    summary: str
    summary_format: str
    translation: str
    translation_language: str
    duration_seconds: float
    created_at: datetime
    updated_at: datetime
    segments: list[TranscriptSegmentResponse] = []

    class Config:
        from_attributes = True


# ------------------------------ AI -----------------------------------------
class SummarizeRequest(BaseModel):
    meeting_id: str
    output_format: str = Field(default="bullets")  # bullets | numbered

    @field_validator("output_format")
    @classmethod
    def _validate_format(cls, v: str) -> str:
        if v not in ("bullets", "numbered"):
            raise ValueError("output_format must be 'bullets' or 'numbered'")
        return v


class SummarizeResponse(BaseModel):
    summary: str
    output_format: str
    engine: str


class TranslateRequest(BaseModel):
    meeting_id: str
    target_language: str  # ISO-ish code, e.g. "es", "fr", "hil", "tl"


class TranslateResponse(BaseModel):
    translation: str
    target_language: str
    language_name: str
    engine: str


TokenResponse.model_rebuild()
