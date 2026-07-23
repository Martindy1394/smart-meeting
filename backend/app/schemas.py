"""Pydantic request/response schemas + validation rules."""
from __future__ import annotations

import json
import re
from datetime import datetime

from pydantic import BaseModel, EmailStr, Field, field_validator

# At least 8 chars, one number and one special character (per requirements).
_PASSWORD_RE = re.compile(r"^(?=.*\d)(?=.*[^A-Za-z0-9]).{8,}$")


# ----------------------------- Auth ----------------------------------------
_USERNAME_RE = re.compile(r"^[A-Za-z0-9._-]{3,64}$")


class SignupRequest(BaseModel):
    first_name: str = Field(..., min_length=1, max_length=128)
    last_name: str = Field(..., min_length=1, max_length=128)
    position: str = Field(..., min_length=1, max_length=128)
    workplace: str = Field(..., min_length=1, max_length=255)
    email: EmailStr  # working email address
    username: str = Field(..., min_length=3, max_length=64)
    password: str

    @field_validator("username")
    @classmethod
    def _validate_username(cls, v: str) -> str:
        v = v.strip()
        if not _USERNAME_RE.match(v):
            raise ValueError(
                "Username must be 3–64 characters and use only letters, "
                "numbers, dots, underscores, or hyphens."
            )
        return v.lower()

    @field_validator("first_name", "last_name", "position", "workplace")
    @classmethod
    def _strip_required(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("This field is required.")
        return v

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
    username: str = Field(..., min_length=1, max_length=64)
    password: str


class ProfileUpdateRequest(BaseModel):
    first_name: str | None = Field(default=None, max_length=128)
    last_name: str | None = Field(default=None, max_length=128)
    position: str | None = Field(default=None, max_length=128)
    workplace: str | None = Field(default=None, max_length=255)
    email: EmailStr | None = None
    username: str | None = Field(default=None, min_length=3, max_length=64)
    password: str | None = None

    @field_validator("username")
    @classmethod
    def _validate_username(cls, v: str | None) -> str | None:
        if v is None:
            return v
        v = v.strip()
        if not _USERNAME_RE.match(v):
            raise ValueError(
                "Username must be 3–64 characters and use only letters, "
                "numbers, dots, underscores, or hyphens."
            )
        return v.lower()

    @field_validator("password")
    @classmethod
    def _validate_password(cls, v: str | None) -> str | None:
        if v is None or v == "":
            return None
        if not _PASSWORD_RE.match(v):
            raise ValueError(
                "Password must be at least 8 characters and include at least "
                "one number and one special character."
            )
        return v


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: "UserResponse"


class UserResponse(BaseModel):
    id: str
    username: str
    email: EmailStr
    first_name: str = ""
    last_name: str = ""
    position: str = ""
    workplace: str = ""
    full_name: str = ""
    created_at: datetime

    class Config:
        from_attributes = True


# --------------------------- Meetings --------------------------------------
class MeetingCreate(BaseModel):
    title: str = Field(default="", max_length=255)
    # ``auto`` = Whisper detects; ASR still biases Hiligaynon via
    # ``whisper_default_language`` (no Spoken language UI).
    language: str = Field(default="auto", max_length=16)
    venue: str = Field(default="", max_length=255)
    meeting_date: datetime | None = None
    attendees: list[str] = Field(default_factory=list)


class MeetingUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=255)
    venue: str | None = Field(default=None, max_length=255)
    meeting_date: datetime | None = None
    attendees: list[str] | None = None
    # Kept for API compatibility; product UI always sends ``auto``.
    language: str | None = Field(default=None, max_length=16)


class TranscriptSegmentResponse(BaseModel):
    id: str
    kind: str
    text: str
    start_time: float
    end_time: float
    seq: int

    class Config:
        from_attributes = True


class LanguageDetectionInfo(BaseModel):
    """Whisper language-detect metadata for analysis when auto-detect fails."""

    language: str | None = None
    confidence: float | None = None
    detected_by: str = ""  # whisper | forced_fallback | live_fallback


class MeetingSummary(BaseModel):
    """Lightweight meeting representation for list / dashboard views."""

    id: str
    title: str
    status: str
    language: str
    language_detection: LanguageDetectionInfo | None = None
    venue: str = ""
    meeting_date: datetime | None = None
    duration_seconds: float
    created_at: datetime
    updated_at: datetime
    # Full BART summary text (used by the Dashboard feed).
    summary: str = ""
    summary_format: str = ""
    translation: str = ""
    translation_language: str = ""
    has_summary: bool = False
    has_translation: bool = False
    has_audio: bool = False
    has_transcript: bool = False

    class Config:
        from_attributes = True


class MeetingDetail(BaseModel):
    id: str
    title: str
    status: str
    language: str
    language_detection: LanguageDetectionInfo | None = None
    venue: str = ""
    meeting_date: datetime | None = None
    attendees: list[str] = []
    final_transcript: str
    summary: str
    summary_format: str
    translation: str
    translation_language: str
    duration_seconds: float
    created_at: datetime
    updated_at: datetime
    has_audio: bool = False
    segments: list[TranscriptSegmentResponse] = []

    class Config:
        from_attributes = True

    @field_validator("attendees", mode="before")
    @classmethod
    def _parse_attendees(cls, v):
        # The ORM stores attendees as a JSON-encoded string.
        if isinstance(v, str):
            try:
                parsed = json.loads(v or "[]")
                return parsed if isinstance(parsed, list) else []
            except (json.JSONDecodeError, TypeError):
                return []
        return v or []


# ------------------------------ AI -----------------------------------------
class SummarizeRequest(BaseModel):
    meeting_id: str
    output_format: str = Field(default="bullets")  # bullets | numbered
    # When true, always re-translate the transcript (ignore cached English).
    force_retranslate: bool = False

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
    # English text produced (or reused) before BART summarization.
    translation: str = ""
    translation_language: str = "English"


class TranslateRequest(BaseModel):
    meeting_id: str
    target_language: str  # ISO-ish code, e.g. "es", "fr", "hil", "tl"


class TranslateResponse(BaseModel):
    translation: str
    target_language: str
    language_name: str
    engine: str


TokenResponse.model_rebuild()
