"""mBART-only helpers for Hiligaynon mistagged / ASR-Tagalogized input.

Stock mBART-50 has no ``hil_XX``. When Whisper captures Ilonggo as Tagalog-ish
noise, ``tl_XX`` decode is the path that actually runs — these helpers stabilize
that encode without changing NLLB/Google/Whisper routing.
"""
from __future__ import annotations

import re

# Distinctive Hiligaynon cues (keep aligned with lang_router markers).
_HIL_CUES = frozenset(
    {
        "gid",
        "indi",
        "sang",
        "kag",
        "amo",
        "subong",
        "subung",
        "naton",
        "pamangkot",
        "maayo",
        "maayong",
        "buwas",
        "dayon",
        "waay",
        "kabalo",
        "nakapoy",
        "kon",
        "tanan",
        "bala",
        "ara",
        "palihog",
        "antes",
        "masunod",
        "magtambong",
        "ilonggo",
        "hiligaynon",
    }
)

# Hiligaynon → Tagalog-leaning cognates so ``tl_XX`` sees more familiar forms.
# Applied only inside the mBART encode path for mistagged / mixed lines.
_HIL_TO_TL_COGNATES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bmaayong\s+aga\b", re.I), "magandang umaga"),
    (re.compile(r"\bmaayong\s+hapon\b", re.I), "magandang hapon"),
    (re.compile(r"\bmaayong\s+gab-i\b", re.I), "magandang gabi"),
    (re.compile(r"\bmaayo\b", re.I), "mabuti"),
    (re.compile(r"\btanan\b", re.I), "lahat"),
    (re.compile(r"\bnaton\b", re.I), "natin"),
    (re.compile(r"\bindi\b", re.I), "hindi"),
    (re.compile(r"\bkag\b", re.I), "at"),
    (re.compile(r"\bsubong\b", re.I), "ngayon"),
    (re.compile(r"\bsubung\b", re.I), "ngayon"),
    (re.compile(r"\bpamangkot\b", re.I), "katanungan"),
    (re.compile(r"\bbala\b", re.I), "ba"),
    (re.compile(r"\bpalihog\b", re.I), "pakiusap"),
    (re.compile(r"\bantes\b", re.I), "bago"),
    (re.compile(r"\bmasunod\b", re.I), "susunod"),
    (re.compile(r"\bkinahanglan\b", re.I), "kailangan"),
    (re.compile(r"\bmiting\b", re.I), "meeting"),
    (re.compile(r"\bnag-atendir\b", re.I), "dumalo"),
    (re.compile(r"\bmagtambong\b", re.I), "dumalo"),
    (re.compile(r"\bara\b", re.I), "mayroon"),
    (re.compile(r"\bsang\b", re.I), "ng"),
    (re.compile(r"\bkon\b", re.I), "kung"),
    (re.compile(r"\bgid\b", re.I), "talaga"),
]


def hiligaynon_cue_ratio(text: str) -> float:
    toks = re.findall(r"[A-Za-zÀ-ÿ']+", (text or "").lower())
    if not toks:
        return 0.0
    hits = sum(1 for t in toks if t in _HIL_CUES)
    return hits / float(len(toks))


def looks_hiligaynon_heavy(text: str, *, min_hits: int = 1, min_ratio: float = 0.04) -> bool:
    toks = re.findall(r"[A-Za-zÀ-ÿ']+", (text or "").lower())
    if not toks:
        return False
    hits = sum(1 for t in toks if t in _HIL_CUES)
    if hits >= 2:
        return True
    return hits >= min_hits and (hits / float(len(toks))) >= min_ratio


def stabilize_hiligaynon_for_tl_mbart(text: str) -> str:
    """Rewrite Hiligaynon cues toward Tagalog cognates for ``tl_XX`` encode."""
    out = (text or "").strip()
    if not out:
        return out
    for pat, repl in _HIL_TO_TL_COGNATES:
        out = pat.sub(repl, out)
    return re.sub(r"\s+", " ", out).strip()
