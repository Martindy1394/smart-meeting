"""AI routes: BART summarization and mBART/NLLB translation via ``invoke_llm``."""
from __future__ import annotations

import json

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
from ..services import action_items, glossary, llm, pipeline_metrics
from ..services.ai_quality import dump_faithfulness

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


def _meeting_glossary(meeting: Meeting) -> list[str]:
    terms = glossary.load_glossary(getattr(meeting, "translation_glossary_json", None))
    # Attendee names are always do-not-translate.
    try:
        from ..services.attendees import load_attendees

        terms = glossary.load_glossary(terms + load_attendees(meeting.attendees))
    except Exception:
        pass
    return terms


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
    gloss = _meeting_glossary(meeting)
    protected_source, gloss_map = glossary.protect(source, gloss)

    cached_english = None
    if not payload.force_retranslate:
        cached_english = (meeting.translation or "").strip() or None
    try:
        with pipeline_metrics.track("summarize"):
            (
                summary,
                summary_engine,
                english,
                translate_engine,
                mt_review,
            ) = llm.summarize_to_english(
                protected_source if not cached_english else source,
                source_language=meeting.language or "auto",
                output_format=payload.output_format,
                existing_english=cached_english,
                source_kind=payload.source_kind or "meeting",
            )
    except llm.LLMUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        )
    english = glossary.restore(english or "", gloss_map)
    engine = f"{translate_engine}+{summary_engine}"
    meeting.summary = summary
    meeting.summary_format = payload.output_format
    if english:
        meeting.translation = english
        meeting.translation_language = "English"

    extractive = "extractive" in (summary_engine or "").lower()
    faith = llm.assess_minutes_faithfulness(summary, english or source)
    xfatih = llm.assess_translation_faithfulness(
        source, english or "", glossary=gloss, review_lines=mt_review
    )
    items = action_items.extract_action_items(summary)

    meeting.extractive_fallback = bool(extractive)
    meeting.faithfulness_json = dump_faithfulness(faith)
    meeting.translation_faithfulness_json = dump_faithfulness(xfatih)
    meeting.action_items_json = json.dumps(items, ensure_ascii=False)
    db.commit()
    return SummarizeResponse(
        summary=summary,
        output_format=payload.output_format,
        engine=engine,
        translation=english or "",
        translation_language="English",
        extractive_fallback=extractive,
        faithfulness=faith,
        translation_faithfulness=xfatih,
        action_items=items,
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
    source = _require_transcript(meeting)
    gloss = _meeting_glossary(meeting)
    protected, gloss_map = glossary.protect(source, gloss)
    try:
        with pipeline_metrics.track("translate"):
            tr = llm.translate(
                protected,
                target_language=payload.target_language,
                source_language=meeting.language or "auto",
            )
            translated, engine = tr.text, tr.engine
            mt_review = list(tr.review_lines or [])
    except llm.LLMUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        )
    translated = glossary.restore(translated, gloss_map)
    xfatih = None
    if (payload.target_language or "").lower() in {"en", "english"}:
        meeting.translation = translated
        meeting.translation_language = "English"
        xfatih = llm.assess_translation_faithfulness(
            source, translated, glossary=gloss, review_lines=mt_review
        )
        meeting.translation_faithfulness_json = dump_faithfulness(xfatih)
        db.commit()
    return TranslateResponse(
        translation=translated,
        target_language=payload.target_language,
        language_name=language_name(payload.target_language),
        engine=engine,
        translation_faithfulness=xfatih,
    )
