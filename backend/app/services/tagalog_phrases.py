"""Exact Tagalog→English meeting phrase lexicon.

Neural MT (NLLB/mBART) cannot guarantee 100% accuracy. For curated board-meeting
phrases we apply deterministic translations first — those covered lines are exact.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

_WS_RE = re.compile(r"\s+")


def _norm(text: str) -> str:
    t = _WS_RE.sub(" ", (text or "").strip().lower())
    t = t.rstrip(".!?…")
    return t


@lru_cache(maxsize=1)
def _phrase_map() -> dict[str, str]:
    """Load seed JSONL + built-in fixtures into a normalized lookup map."""
    mapping: dict[str, str] = {}
    # Prefer repo seed next to this package (dev) or cwd-relative path.
    candidates = [
        Path(__file__).resolve().parents[3] / "scripts/ph_mt/seed_tagalog_en.jsonl",
        Path("scripts/ph_mt/seed_tagalog_en.jsonl"),
        Path("/workspace/scripts/ph_mt/seed_tagalog_en.jsonl"),
    ]
    for path in candidates:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            src = _norm(str(row.get("src") or row.get("source") or ""))
            tgt = (row.get("tgt") or row.get("reference") or "").strip()
            if src and tgt:
                mapping[src] = tgt
        break
    return mapping


def lookup_exact(text: str) -> str | None:
    """Return exact English if the whole clause matches a curated Tagalog phrase."""
    key = _norm(text)
    if not key:
        return None
    hit = _phrase_map().get(key)
    if hit:
        return hit
    # Also try with trailing period restored for seeds that include punctuation.
    return _phrase_map().get(key + ".")


def lexicon_size() -> int:
    return len(_phrase_map())
