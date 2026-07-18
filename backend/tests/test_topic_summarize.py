"""Unit tests for topic-aware BART summarization helpers (no model weights)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services import llm


def test_segment_transcript_topics_respects_token_budget():
    # Distinct units so idea-unit dedupe does not collapse them.
    topics = [
        "Finance reviewed quarterly payroll, vendor invoices, and travel reimbursements carefully.",
        "Facilities inspected the auditorium lighting, stage curtains, and emergency exits thoroughly.",
        "Marketing drafted social posts, email newsletters, and radio advertisements for launch week.",
        "Legal examined contract clauses, liability waivers, and insurance coverage for partners.",
        "Catering confirmed menu options, dietary restrictions, and delivery timelines for guests.",
        "Security assigned entrance badges, parking attendants, and overnight patrol schedules.",
        "IT prepared livestream encoders, backup microphones, and wifi capacity for attendees.",
        "Education outlined workshop modules, mentoring sessions, and certificate requirements.",
    ]
    text = " ".join(topics * 3)
    chunks = llm.segment_transcript_topics(text, max_tokens=80, min_units=1)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert llm._estimate_tokens(chunk) <= 95


def test_segment_transcript_topics_splits_on_low_similarity():
    topic_a = (
        "The microphone sound check went well. Everybody loves the opening song lyrics. "
        "The band practiced the chorus again."
    )
    topic_b = (
        "Bakit ba lagi na lang ang mga Pilipino ay naging bobo sa pagpili ng politiko. "
        "Dapat pag-isipan ang sistema ng eleksyon sa bansa."
    )
    text = f"{topic_a} {topic_b}"
    chunks = llm.segment_transcript_topics(
        text,
        max_tokens=960,
        similarity_threshold=0.15,
        min_units=1,
    )
    assert len(chunks) >= 2
    joined = " ".join(chunks)
    assert "microphone" in joined.lower() or "sound" in joined.lower()
    assert "politiko" in joined.lower() or "pilipino" in joined.lower()


def test_segment_keeps_speaker_turns_intact():
    text = (
        "Alice: We should approve the venue budget today.\n"
        "Bob: I agree, and we also need to finalize the catering.\n"
        "Carol: Let's move the marketing discussion to next week."
    )
    chunks = llm.segment_transcript_topics(text, max_tokens=960, min_units=2)
    assert chunks
    # Speaker turn bodies should appear without mid-turn cuts for short turns.
    assert "approve the venue budget" in chunks[0]


def test_summary_sentences_to_units_makes_bullets():
    raw = (
        "The team approved the budget. Marketing will launch next week. "
        "Facilities will prepare the hall."
    )
    units = llm._summary_sentences_to_units(raw)
    assert len(units) >= 3
    formatted = llm._format_summary(units, "bullets")
    assert formatted.startswith("- ")
    assert formatted.count("\n") >= 2


def test_format_topic_summaries_with_headers():
    sections = [
        ("Opening", ["Mic check completed.", "Everyone was ready."]),
        ("Decisions", ["Budget was approved."]),
    ]
    out = llm._format_topic_summaries(sections, "bullets")
    assert "Opening" in out
    assert "Decisions" in out
    assert "- Mic check completed." in out
    assert "- Budget was approved." in out


def test_short_summarize_stays_idea_preserving():
    text = (
        "Mic test sound check. Everybody loves the things you do. "
        "Bakit ba lagi na lang ang mga Pilipino ay naging bobo sa pagpili ng politiko."
    )
    summary, engine = llm.summarize(text, output_format="bullets")
    assert engine == "bart-meeting-minutes"
    assert summary.startswith("- ")
    assert "Mic" in summary or "sound" in summary.lower()


def test_english_source_kind_summarizes_in_english():
    text = (
        "The board approved the quarterly budget after review. "
        "Marketing will launch the campaign next week. "
        "Facilities will prepare the hall and stage lighting. "
        "Security assigned entrance badges for all guests."
    )
    summary, engine = llm.summarize(
        text, output_format="bullets", source_kind="english_translation"
    )
    assert "english" in engine
    assert summary.startswith("- ")
    assert "budget" in summary.lower() or "Marketing" in summary or "campaign" in summary.lower()


def test_english_covers_transcript_helper():
    assert llm._english_covers_transcript(
        "The team approved the budget and scheduled the launch next week for everyone.",
        "The team approved the budget and scheduled the launch next week for everyone present.",
    )
    assert not llm._english_covers_transcript("Hi", "long transcript " * 40)


def test_summarize_to_english_reuses_cached_translation(monkeypatch=None):
    english = (
        "The committee approved the venue budget. "
        "Catering confirmed the menu for guests. "
        "IT prepared livestream equipment for attendees. "
        "Education outlined workshop modules for mentors."
    )
    summary, engine, out_en, tr_engine = llm.summarize_to_english(
        english,
        source_language="en",
        output_format="bullets",
        existing_english=english,
    )
    assert tr_engine == "cached-english"
    assert out_en == english or out_en.startswith("The committee")
    assert summary.startswith("- ")
    assert "english" in engine or engine.startswith("bart")


if __name__ == "__main__":
    test_segment_transcript_topics_respects_token_budget()
    test_segment_transcript_topics_splits_on_low_similarity()
    test_segment_keeps_speaker_turns_intact()
    test_summary_sentences_to_units_makes_bullets()
    test_format_topic_summaries_with_headers()
    test_short_summarize_stays_idea_preserving()
    test_english_source_kind_summarizes_in_english()
    test_english_covers_transcript_helper()
    test_summarize_to_english_reuses_cached_translation()
    print("all_topic_summarize_tests_passed")
