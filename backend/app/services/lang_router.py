"""Three-way line language router for PH meeting transcripts.

Detects English / Tagalog / Hiligaynon per line (or idea unit) using lexical
wordlists — **not** langdetect / CLD, which routinely mislabel Hiligaynon as
Tagalog or Cebuano because of shared Visayan/PH vocabulary.

Returns an uncertainty flag when Hiligaynon ↔ Tagalog ↔ Cebuano scores are
too close for a confident automatic choice (common in Panay / Negros border
speech).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

RouteLang = Literal["en", "tl", "hil", "unknown"]

# Distinctive Hiligaynon / Ilonggo markers (avoid Tagalog-only and bare shared
# particles like ``sa`` / ``na`` / ``ang`` that appear across PH languages).
_HILIGAYNON_MARKERS = frozenset(
    {
        "gid",
        "guid",
        "indi",
        "buwas",
        "buas",
        "nakapoy",
        "mangita",
        "kabalo",
        "subong",
        "subung",
        "dayon",
        "naton",
        "ninyo",
        "kamo",
        "inyo",
        "ila",
        "kon",
        "kag",
        "sang",
        "amo",
        "guin",
        "gin",
        "magapangita",
        "magkadto",
        "kadto",
        "diri",
        "dida",
        "didto",
        "ara",
        "waay",
        "wala",  # shared but common with gid/indi collocations
        "ti",
        "rang",
        "run",
        "ron",
        "gali",
        "abi",
        "balay",
        "kaon",
        "siling",
        "haman",
        "pahulay",
        "pamangkot",
        "sabat",
        "himuon",
        "himua",
        "maghimu",
        "maayo",
        "maayong",
        "aga",
        "hapon",
        "gab-i",
        "tukar",
        "pulong",
        "hambal",
        "hambalon",
        "diskusyon",
        "miting",
        "ilonggo",
        "hiligaynon",
    }
)

# Tagalog-preferring markers (minimize overlap with distinctive Hiligaynon).
_TAGALOG_MARKERS = frozenset(
    {
        "ang",
        "mga",
        "nang",
        "ng",
        "yung",
        "yun",
        "iyong",
        "iyan",
        "iyon",
        "ito",
        "hindi",
        "huwag",
        "kasi",
        "kaya",
        "bakit",
        "kapag",
        "kung",  # also hil — weighted lower via hil-only boosts
        "po",
        "opo",
        "ho",
        "naman",
        "lang",
        "pala",
        "daw",
        "raw",
        "ba",
        "ay",
        "natin",
        "namin",
        "nila",
        "niya",
        "siya",
        "tayo",
        "kami",
        "kayo",
        "ako",
        "ikaw",
        "meron",
        "mayroon",
        "dapat",
        "sana",
        "talaga",
        "ganun",
        "ganon",
        "ngayon",
        "dito",
        "doon",
        "roon",
        "pangarap",
        "ibigin",
        "habang",
        "panahon",
        "buhay",
        "salamat",
        "mahal",
        "gusto",
        "kailangan",
        "pwede",
        "puwede",
        "sige",
        "tagalog",
        "filipino",
    }
)

_ENGLISH_MARKERS = frozenset(
    {
        "the",
        "and",
        "you",
        "that",
        "with",
        "this",
        "have",
        "from",
        "they",
        "were",
        "was",
        "are",
        "is",
        "my",
        "your",
        "our",
        "when",
        "because",
        "will",
        "would",
        "should",
        "could",
        "about",
        "into",
        "their",
        "there",
        "which",
        "what",
        "where",
        "while",
        "after",
        "before",
        "meeting",
        "board",
        "decision",
        "action",
        "please",
        "thank",
        "thanks",
        "everyone",
        "today",
        "tomorrow",
        "currently",
        "especially",
    }
)

# Multi-word Hiligaynon collocations — strong signal even with shared particles.
_HIL_COLLOCATIONS = (
    "wala gid",
    "indi gid",
    "waay gid",
    " amo ni",
    "sang mga",
    "kag ang",
    "kon indi",
    "maayong aga",
    "maayong hapon",
    "subong nga",
    "nakapoy na",
)

_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ']+")


@dataclass(frozen=True)
class LineRoute:
    """Per-line routing decision for the three-way MT splitter."""

    language: RouteLang
    confidence: float
    uncertain: bool
    scores: dict[str, float]
    reason: str = ""


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _marker_ratio(tokens: list[str], markers: frozenset[str]) -> float:
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in markers)
    return hits / float(len(tokens))


def _collocation_boost(text: str) -> float:
    low = f" {(text or '').lower()} "
    hits = sum(1 for c in _HIL_COLLOCATIONS if c in low)
    return min(0.35, 0.12 * hits)


def classify_line(
    text: str,
    *,
    meeting_language: str | None = None,
) -> LineRoute:
    """Classify one transcript line/unit as en / tl / hil (or unknown).

    ``uncertain=True`` when Hiligaynon vs Tagalog scores are too close — callers
    should still pick a best-effort route but flag the line for manual review.
    """
    raw = (text or "").strip()
    tokens = _tokens(raw)
    if not tokens:
        return LineRoute("unknown", 0.0, True, {}, "empty")

    en = _marker_ratio(tokens, _ENGLISH_MARKERS)
    tl = _marker_ratio(tokens, _TAGALOG_MARKERS)
    hil = _marker_ratio(tokens, _HILIGAYNON_MARKERS) + _collocation_boost(raw)

    # Hiligaynon-only markers outweigh shared particles.
    hil_only = sum(
        1
        for t in tokens
        if t in _HILIGAYNON_MARKERS and t not in _TAGALOG_MARKERS
    )
    if hil_only:
        hil += min(0.25, 0.08 * hil_only)

    scores = {"en": round(en, 4), "tl": round(tl, 4), "hil": round(hil, 4)}

    # Clear English: English markers dominate and PH signal is weak.
    if en >= 0.10 and en >= tl + 0.04 and en >= hil + 0.04 and max(tl, hil) < 0.10:
        return LineRoute("en", min(1.0, en + 0.2), False, scores, "english_markers")

    # Strong Hiligaynon.
    if hil >= 0.10 and hil >= tl + 0.05:
        return LineRoute("hil", min(1.0, hil + 0.15), False, scores, "hiligaynon_markers")

    # Strong Tagalog.
    if tl >= 0.10 and tl >= hil + 0.05:
        return LineRoute("tl", min(1.0, tl + 0.15), False, scores, "tagalog_markers")

    # Ambiguous PH — close hil/tl (or both weak).
    ph_top = max(hil, tl)
    gap = abs(hil - tl)
    meeting = (meeting_language or "").strip().lower()
    uncertain = ph_top >= 0.06 and gap < 0.06

    if hil >= tl and hil >= 0.06:
        lang: RouteLang = "hil"
        conf = hil
        reason = "hil_lean_close" if uncertain else "hil_lean"
    elif tl >= 0.06:
        lang = "tl"
        conf = tl
        reason = "tl_lean_close" if uncertain else "tl_lean"
    elif en >= 0.06:
        lang = "en"
        conf = en
        reason = "english_weak"
        uncertain = False
    else:
        # Fall back to meeting language bias when lexical evidence is thin.
        if meeting in {"hil", "hiligaynon", "ilonggo"}:
            lang, conf, reason = "hil", 0.35, "meeting_bias_hil"
            uncertain = True
        elif meeting in {"tl", "tagalog", "fil", "filipino"}:
            lang, conf, reason = "tl", 0.35, "meeting_bias_tl"
            uncertain = True
        elif en >= tl and en >= hil:
            lang, conf, reason = "en", max(en, 0.2), "default_english"
            uncertain = True
        else:
            lang, conf, reason = "unknown", 0.2, "no_signal"
            uncertain = True

    # Border-case: both hil and tl present with tiny gap → always review.
    if gap < 0.04 and min(hil, tl) >= 0.06:
        uncertain = True
        reason = "hil_tl_ambiguous"

    return LineRoute(lang, float(min(1.0, conf)), uncertain, scores, reason)


def route_units(
    units: list[str],
    *,
    meeting_language: str | None = None,
) -> list[tuple[str, LineRoute]]:
    """Classify each unit; preserves input order for reassembly."""
    return [(u, classify_line(u, meeting_language=meeting_language)) for u in units]
