"""Regression tests for lyric/song MT content-loss fixes (Issues 1–5)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services import llm

# Punctuated form of the reproduction verse — what dedupe must preserve.
LYRIC_UNITS = [
    "Pumapatak sa mga halang at mga bulaklak.",
    "Maari bang huwag ka na.",
    "Sa piling ko'y lumisan pa.",
    "Hatid ko lamang ang saksihin.",
    "Buhus pa ulan.",
    "Aking mundo'y lunureng tuluyan.",
    "Hatid mo may bagyo dalangin ito ng puso ko ang lungiliyap.",
    "Damdamin ko'y maapaw.",
    "Pag-ibig ko'y humihiyaw.",
    "Sa tuwa tuwing mong ulan at tupil-ilin ka.",
]

MISSING_MARKERS = [
    "Buhus pa ulan",
    "Hatid mo may bagyo",
    "Damdamin ko'y maapaw",
    "humihiyaw",
]


def test_issue1_diagnostic_units_survive_dedupe_on_punctuated_lyrics():
    """Issue 1/2: thematic lyric lines must remain after dedupe."""
    post = llm._dedupe_units(list(LYRIC_UNITS))
    assert len(post) == len(LYRIC_UNITS)
    joined = " ".join(post)
    for marker in MISSING_MARKERS:
        assert marker.lower() in joined.lower(), marker


def test_dedupe_still_collapses_adjacent_asr_repetition():
    """Meeting-speech ASR loops must still be collapsed."""
    units = [
        "We need to approve the quarterly budget today.",
        "We need to approve the quarterly budget today.",
        "We need to approve the quarterly budget today please.",
        "Facilities will open the auditorium at noon.",
    ]
    post = llm._dedupe_units(units)
    assert len(post) == 2
    assert "budget" in post[0].lower()
    assert "auditorium" in post[1].lower()


def test_dedupe_local_window_keeps_distant_thematic_echo():
    """Thematic echo far from the original line must not be killed."""
    units = [
        "Ulan sa hardin ng mga bulaklak ngayong umaga.",
        "Ang komite ay nagkaisa sa usaping badyet kanina.",
        "Tapos ay tinapos ang botohan nang maayos.",
        "Ulan sa hardin ng mga bulaklak ngayong gabi.",  # thematic echo, not adjacent ASR
    ]
    post = llm._dedupe_units(units)
    assert len(post) == 4


def test_garbage_detector_catches_honor_m_swear():
    """Issue 4 reproduction pair — must not pass the gate."""
    assert llm._is_garbage_english_translation(
        "sa mga halang at mga bulaklak",
        "To honor m Swear",
    )


def test_garbage_detector_allows_known_good_short_translations():
    goods = [
        ("Salamat sa inyong oras.", "Thank you for your time."),
        (
            "Si Maria ang magiging responsible sa follow-up.",
            "Maria will be responsible for the follow-up.",
        ),
        (
            "Mayroon bang katanungan tungkol sa agenda?",
            "Are there any questions about the agenda?",
        ),
    ]
    for src, hyp in goods:
        assert not llm._is_garbage_english_translation(src, hyp), (src, hyp)


def test_mark_untranslated_and_strip_for_downstream():
    marked = llm.mark_untranslated("Pagmahalang mga m")
    assert marked.startswith("[untranslated:")
    assert "Pagmahalang" in marked
    assert llm.iter_untranslated_spans(f"Hello {marked} world") == [
        "Pagmahalang mga m"
    ]
    assert "Pagmahalang" not in llm.strip_untranslated_spans(
        f"Hello {marked} world", keep_inner=False
    )
    assert "Pagmahalang" in llm.strip_untranslated_spans(
        f"Hello {marked} world", keep_inner=True
    )


def test_kept_source_is_marked_in_translate_path():
    """Issue 3: failed MT must not splice raw source unmarked."""

    def boom(*a, **k):
        raise llm._NonEnglishTranslation("x")

    with (
        patch.object(llm, "_translate_unit_with_context", side_effect=boom),
        patch.object(llm, "_segment_idea_units", return_value=["Kailangan nating umalis ngayon kasi late na."]),
        patch.object(llm, "_normalize_spoken_transcript", side_effect=lambda t: t),
    ):
        # Avoid hallucination import path clearing the unit.
        with patch.dict("sys.modules", {"app.services.transcription": None}):
            tr = llm._translate_to_english(
                "Kailangan nating umalis ngayon kasi late na.", "tl"
            )
    assert "[untranslated:" in tr.text
    assert any(r.get("section") == "Untranslated" for r in tr.review_lines)


def test_coverage_review_on_severe_mass_loss():
    """Issue 5: low PH→EN mass coverage lands in review_lines."""

    def tiny_eng(unit, prev, src, **k):
        return "Yes."

    long_unit = (
        "Sa mga 11111111111111111halang at mga bulaklak Maari bang huwag ka na "
        "sa piling ko lumisan pa hatid ko lamang ang saksihin buhus pa ulan "
        "aking mundo lunureng tuluyan hatid mo may bagyo dalangin ito ng puso"
    )
    # Ensure content words exist (no digits-only tokens).
    long_unit = (
        "Sa mga halang at mga bulaklak Maari bang huwag ka na sa piling ko "
        "lumisan pa hatid ko lamang ang saksihin buhus pa ulan aking mundo "
        "lunureng tuluyan hatid mo may bagyo dalangin ito ng puso ko"
    )
    with (
        patch.object(llm, "_translate_unit_with_context", side_effect=tiny_eng),
        patch.object(llm, "_segment_idea_units", return_value=[long_unit]),
        patch.object(llm, "_normalize_spoken_transcript", side_effect=lambda t: t),
        patch.object(llm, "_is_garbage_english_translation", return_value=False),
        patch.object(llm, "_looks_like_latin_script", return_value=True),
    ):
        tr = llm._translate_to_english(long_unit, "tl")
    assert any(r.get("section") == "Coverage" for r in tr.review_lines)


def test_faithfulness_flags_untranslated_section():
    text = "Hello. " + llm.mark_untranslated("Raw Tagalog clause here")
    faith = llm.assess_translation_faithfulness(
        "Raw Tagalog clause here", text
    )
    assert any(u.get("section") == "Untranslated" for u in faith["untraced"])


if __name__ == "__main__":
    test_issue1_diagnostic_units_survive_dedupe_on_punctuated_lyrics()
    test_dedupe_still_collapses_adjacent_asr_repetition()
    test_dedupe_local_window_keeps_distant_thematic_echo()
    test_garbage_detector_catches_honor_m_swear()
    test_garbage_detector_allows_known_good_short_translations()
    test_mark_untranslated_and_strip_for_downstream()
    test_kept_source_is_marked_in_translate_path()
    test_coverage_review_on_severe_mass_loss()
    test_faithfulness_flags_untranslated_section()
    print("all_lyric_mt_pipeline_tests_passed")
