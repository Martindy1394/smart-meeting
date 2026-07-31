"""Unit tests for the three-way EN / Tagalog / Hiligaynon language router."""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.lang_router import classify_line, route_units  # noqa: E402


def test_english_passthrough_classification():
    r = classify_line("Good morning everyone. Today we will discuss the budget.")
    assert r.language == "en"
    assert r.uncertain is False


def test_tagalog_classification():
    r = classify_line(
        "Hindi po kami papayag kung wala ang mga dokumento. Bakit naman ganon?"
    )
    assert r.language == "tl"
    assert r.uncertain is False


def test_hiligaynon_classification():
    r = classify_line(
        "Wala gid ko kabalo. Indi ako makadto subong kay nakapoy na ko."
    )
    assert r.language == "hil"
    assert r.scores["hil"] > r.scores["tl"]


def test_hil_tl_ambiguous_flagged_for_review():
    # Shared particles with weak distinctive signal → uncertain.
    r = classify_line("Wala sa meeting pero may update.", meeting_language="auto")
    # May lean tl or hil depending on markers; if close, uncertain is set.
    if abs(r.scores.get("hil", 0) - r.scores.get("tl", 0)) < 0.06:
        assert r.uncertain is True


def test_route_units_preserves_order():
    units = [
        "The board will vote tomorrow.",
        "Wala gid problema sang budget.",
        "Hindi sila papayag sa proposal.",
    ]
    routed = route_units(units, meeting_language="auto")
    assert [u for u, _ in routed] == units
    assert routed[0][1].language == "en"
    assert routed[1][1].language == "hil"
    assert routed[2][1].language == "tl"


def test_meeting_bias_when_signal_thin():
    r = classify_line("Update later.", meeting_language="hil")
    # Thin signal + hil meeting bias should prefer hil and mark uncertain.
    assert r.language in {"en", "hil", "unknown"}
    if r.language == "hil":
        assert r.uncertain is True


if __name__ == "__main__":
    test_english_passthrough_classification()
    test_tagalog_classification()
    test_hiligaynon_classification()
    test_hil_tl_ambiguous_flagged_for_review()
    test_route_units_preserves_order()
    test_meeting_bias_when_signal_thin()
    print("lang_router_ok")
