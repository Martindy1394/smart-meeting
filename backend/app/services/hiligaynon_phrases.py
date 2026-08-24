"""Exact Hiligaynon→English meeting phrase lexicon (mBART path reliability).

Stock mBART has no ``hil_XX`` token; curated lines short-circuit before the
degraded ``id_ID`` / mistagged ``tl_XX`` decode.
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
    mapping: dict[str, str] = {}
    candidates = [
        Path(__file__).resolve().parents[3] / "scripts/ph_mt/seed_hiligaynon_en.jsonl",
        Path("scripts/ph_mt/seed_hiligaynon_en.jsonl"),
        Path("/workspace/scripts/ph_mt/seed_hiligaynon_en.jsonl"),
        Path(__file__).resolve().parents[3]
        / "scripts/ph_mt/fixtures/hiligaynon_en_sample.jsonl",
        Path("scripts/ph_mt/fixtures/hiligaynon_en_sample.jsonl"),
        Path(__file__).resolve().parents[3]
        / "scripts/ph_mt/fixtures/hiligaynon_as_tagalog_noise.jsonl",
        Path("scripts/ph_mt/fixtures/hiligaynon_as_tagalog_noise.jsonl"),
    ]
    seen_paths: set[str] = set()
    for path in candidates:
        key = str(path.resolve()) if path.is_file() else ""
        if not key or key in seen_paths:
            continue
        seen_paths.add(key)
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
    return mapping


def lookup_exact(text: str) -> str | None:
    """Return exact English if the whole clause matches a curated Hiligaynon phrase."""
    key = _norm(text)
    if not key:
        return None
    hit = _phrase_map().get(key)
    if hit:
        return hit
    return _phrase_map().get(key + ".")


def lexicon_size() -> int:
    return len(_phrase_map())
