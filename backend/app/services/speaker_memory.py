"""Operator corrections for speaker identity (adaptive aliases).

Stores per-owner JSON under ``data/speaker_memory/``. When a reviewer maps
Voice N or a mis-heard introduction to a canonical roster name, later meetings
reuse that alias. This is not a neural embedding store.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger("smart_meeting.speaker_memory")


def _dir() -> Path:
    try:
        from ..config import settings

        root = Path(getattr(settings, "audio_storage_dir", "./data/audio") or "./data/audio")
        base = root.parent / "speaker_memory"
    except Exception:
        base = Path("./data/speaker_memory")
    base.mkdir(parents=True, exist_ok=True)
    return base


def _path(owner_id: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in (owner_id or "anon"))
    return _dir() / f"{safe}.json"


def load_aliases(owner_id: str) -> dict[str, str]:
    path = _path(owner_id)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        logger.exception("Could not read speaker memory %s", path)
        return {}
    raw = data.get("aliases") if isinstance(data, dict) else {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        key = str(k or "").strip().casefold()
        val = str(v or "").strip()
        if key and val:
            out[key] = val
    return out


def remember_alias(owner_id: str, heard: str, canonical: str) -> None:
    heard_k = (heard or "").strip().casefold()
    canon = (canonical or "").strip()
    if not heard_k or not canon:
        return
    path = _path(owner_id)
    aliases = load_aliases(owner_id)
    aliases[heard_k] = canon
    payload = {"aliases": aliases}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def remember_voice(owner_id: str, meeting_id: str, speaker_index: int, canonical: str) -> None:
    """Bind Voice N in a meeting to a name; also alias the voice label."""
    canon = (canonical or "").strip()
    if not canon:
        return
    remember_alias(owner_id, f"voice {int(speaker_index or 0)}", canon)
    remember_alias(owner_id, f"{meeting_id}:{int(speaker_index or 0)}", canon)
