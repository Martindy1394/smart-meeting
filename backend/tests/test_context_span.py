"""Tests for context-window span extraction safeguards (no model weights)."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services import llm


def _long_unit() -> str:
    # ≥ 6 words so context-window path is eligible.
    return (
        "Ang komite ay nagkaisa na aprubahan ang buong taunang badyet "
        "para sa proyekto ng paaralan"
    )


def test_short_unit_skips_context_entirely():
    """Units below the word threshold translate alone (no window call)."""
    unit = "Aprubahan ang badyet ngayon"  # 4 words
    prev = [
        "Kanina pinag-usapan nila ang venue at oras ng programa nang detalyado.",
        "Pagkatapos ay sinuri ang listahan ng mga dadalo sa pulong.",
    ]
    calls: list[str] = []

    def fake_nllb(text: str, source_language: str = "auto") -> str:
        calls.append(text)
        return f"EN({text.split()[0]})"

    with patch.object(llm, "_nllb_translate_to_english", side_effect=fake_nllb):
        out = llm._translate_unit_with_context(
            unit, prev, "tl", engine="nllb"
        )

    assert len(calls) == 1
    assert calls[0] == unit
    assert out == "EN(Aprubahan)"


def test_long_unit_stable_context_uses_span_extraction():
    """Stable sentence counts keep context and return a non-empty trailing span."""
    unit = (
        "Ang komite ay nagkaisa na aprubahan ang buong taunang badyet "
        "para sa proyekto"
    )  # 11 words — above short-unit threshold
    prev = [
        (
            "Una naming tinalakay ang lokasyon ng pulong sa umaga kasama ang "
            "mga detalye ng silid at upuan."
        ),
        (
            "Sunod ay sinuri ang mga dokumento ng komite nang maingat bago "
            "ang botohan."
        ),
    ]
    expected_tail = (
        "The committee agreed to approve the full annual project budget."
    )
    calls: list[str] = []

    def fake_nllb(text: str, source_language: str = "auto") -> str:
        calls.append(text)
        if text == unit:
            return expected_tail
        # Mirror source sentence boundaries (3 clauses → 3 English sentences).
        return (
            "First we discussed the meeting room location and seating details. "
            "Next the committee documents were reviewed carefully before the vote. "
            + expected_tail
        )

    with patch.object(llm, "_nllb_translate_to_english", side_effect=fake_nllb):
        out = llm._translate_unit_with_context(
            unit, prev, "tl", engine="nllb"
        )

    assert out
    assert out == expected_tail
    # Context window was used (not only the unit-alone path).
    assert any(c != unit for c in calls)
    assert "First we discussed" not in out
    assert not llm._span_looks_truncated(out, unit)


def test_sentence_count_divergence_falls_back_to_unit_alone():
    """Merged English sentences (parity < 0.5x) discard the window trim."""
    unit = _long_unit()
    # Five short punctuated source clauses in the window.
    prev = [
        "Una. Pangalawa. Pangatlo.",
        "Pang-apat. Ikalima.",
    ]
    alone = "The committee alone approved the annual school project budget today."
    calls: list[str] = []

    def fake_nllb(text: str, source_language: str = "auto") -> str:
        calls.append(text)
        if text == unit:
            return alone
        # Several source sentences collapse into one English sentence.
        return (
            "First second third fourth fifth and the committee approved "
            "the full annual school project budget in one breath"
        )

    with patch.object(llm, "_nllb_translate_to_english", side_effect=fake_nllb):
        out = llm._translate_unit_with_context(
            unit, prev, "tl", engine="nllb"
        )

    assert out == alone
    assert any(c == unit for c in calls), "expected context-free retry"
    # Window was attempted first.
    assert any(c != unit for c in calls)


def test_empty_span_falls_back_to_nonempty_unit_translation():
    """Empty extract must not propagate; unit-alone translation is returned."""
    unit = _long_unit()
    prev = ["Una naming tinalakay ang lokasyon ng pulong sa umaga nang detalyado."]
    alone = "The committee agreed to approve the annual budget for the school project."

    def fake_nllb(text: str, source_language: str = "auto") -> str:
        if text == unit:
            return alone
        return (
            "Earlier they discussed the meeting venue in detail this morning. "
            "Then the board moved to the next agenda item carefully."
        )

    with (
        patch.object(llm, "_nllb_translate_to_english", side_effect=fake_nllb),
        patch.object(llm, "_extract_target_span", return_value=""),
    ):
        out = llm._translate_unit_with_context(
            unit, prev, "tl", engine="nllb"
        )

    assert out == alone
    assert out.strip() != ""


def test_truncated_span_falls_back_to_unit_alone():
    """Suspiciously short extracted span triggers a context-free retry."""
    unit = _long_unit()
    prev = [
        "Kanina ay mahaba ang talakayan tungkol sa venue oras at dadalo.",
        "Pagkatapos ay sinuri ang dokumento ng bawat departamento nang detalyado.",
    ]
    alone = (
        "The committee unanimously agreed to approve the full annual "
        "budget for the school project."
    )

    def fake_nllb(text: str, source_language: str = "auto") -> str:
        if text == unit:
            return alone
        return (
            "Earlier they discussed the venue schedule and attendees. "
            "Then each department document was reviewed carefully. "
            "The committee agreed to approve the full annual school project budget."
        )

    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Capture()
    handler.setLevel(logging.INFO)
    log = logging.getLogger("smart_meeting.llm")
    prev_level = log.level
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    try:
        with (
            patch.object(llm, "_nllb_translate_to_english", side_effect=fake_nllb),
            patch.object(llm, "_extract_target_span", return_value="Yes."),
            patch.object(llm, "_sentence_count_parity_ok", return_value=True),
        ):
            out = llm._translate_unit_with_context(
                unit, prev, "tl", engine="nllb"
            )
    finally:
        log.removeHandler(handler)
        log.setLevel(prev_level)

    assert out == alone
    assert any("truncated span" in r.getMessage() for r in records)


def test_google_engine_still_skips_context():
    """Hiligaynon Google path stays line-local (out of scope for this fix)."""
    unit = "Maganda ang adlaw subong para sa miting"
    prev = ["Sang una ginbinagbinag namon ang venue."]
    called: list[str] = []

    def fake_google(text: str) -> str:
        called.append(text)
        return "It is a nice day today for the meeting"

    import app.services.google_translate as gt

    with patch.object(gt, "translate_hiligaynon_to_english", side_effect=fake_google):
        out = llm._translate_unit_with_context(
            unit, prev, "hil", engine="google"
        )

    assert called == [unit]
    assert "nice day" in out.lower()


def test_extract_target_span_prefers_trailing_context():
    full = "Earlier they discussed venues. Then the board approved the budget today."
    span = llm._extract_target_span(
        full,
        "the board approved the budget today",
        "Earlier they discussed venues. the board approved the budget today",
    )
    assert "approved" in span.lower()


def test_sentence_count_parity_helper():
    assert llm._sentence_count_parity_ok(
        "One sentence here. Two sentence here.",
        "One English sentence. Two English sentence.",
    )
    assert not llm._sentence_count_parity_ok(
        "A. B. C. D. E.",
        "Everything merged into one long English sentence without breaks",
    )


if __name__ == "__main__":
    test_short_unit_skips_context_entirely()
    test_long_unit_stable_context_uses_span_extraction()
    test_sentence_count_divergence_falls_back_to_unit_alone()
    test_empty_span_falls_back_to_nonempty_unit_translation()
    test_truncated_span_falls_back_to_unit_alone()
    test_google_engine_still_skips_context()
    test_extract_target_span_prefers_trailing_context()
    test_sentence_count_parity_helper()
    print("all_context_span_tests_passed")
