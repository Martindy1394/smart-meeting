"""AI routes: BART summarization and mBART translation via ``invoke_llm``."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..database import get_db
from ..deps import get_current_user
from ..languages import language_name, list_languages
from ..limiter import ai_rate_limit
from ..models import Meeting, User
from ..routers.meetings import _get_owned_meeting
from ..schemas import (
    SummarizeRequest,
    SummarizeResponse,
    TranslateRequest,
    TranslateResponse,
)
from ..services import llm

router = APIRouter(prefix="/api/ai", tags=["ai"])


@router.get("/languages")
def get_languages():
    return list_languages()


def _require_transcript(meeting: Meeting) -> str:
    text = (meeting.final_transcript or "").strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This meeting has no finalized transcript to process yet.",
        )
    return text


@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    dependencies=[Depends(ai_rate_limit())],
)
def summarize(
    payload: SummarizeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    meeting = _get_owned_meeting(payload.meeting_id, current_user, db)
    source = _require_transcript(meeting)
    # Full transcript → English (mBART) → contextual BART summary.
    try:
        summary, summary_engine, english, translate_engine = llm.summarize_to_english(
            source,
            source_language=meeting.language or "auto",
            output_format=payload.output_format,
            existing_english=(meeting.translation or "").strip() or None,
        )
    except llm.LLMUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        )
    engine = f"{translate_engine}+{summary_engine}"
    meeting.summary = summary
    meeting.summary_format = payload.output_format
    if english:
        meeting.translation = english
        meeting.translation_language = "English"
    db.commit()
    return SummarizeResponse(
        summary=summary,
        output_format=payload.output_format,
        engine=engine,
        translation=english or "",
        translation_language="English",
    )


@router.post(
    "/translate",
    response_model=TranslateResponse,
    dependencies=[Depends(ai_rate_limit())],
)
def translate(
    payload: TranslateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    meeting = _get_owned_meeting(payload.meeting_id, current_user, db)
    transcript = _require_transcript(meeting)
    try:
        translation, engine = llm.invoke_llm(
            "translate",
            transcript,
            target_language=payload.target_language,
            # Match summarize: auto-detect / PH → id_ID English path.
            source_language=meeting.language or "auto",
        )
    except llm.LLMUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    name = language_name(payload.target_language)
    meeting.translation = translation
    meeting.translation_language = name
    db.commit()
    return TranslateResponse(
        translation=translation,
        target_language=payload.target_language,
        language_name=name,
        engine=engine,
    )
