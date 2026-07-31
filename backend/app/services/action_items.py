"""Structured Action Item extraction (owner / action / due_date) on BART prose."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any


_OWNER_PATTERNS = [
    re.compile(
        r"^(?P<owner>[A-Z][\w.'-]+(?:\s+[A-Z][\w.'-]+){0,2})\s+"
        r"(?:will|shall|to|must|should)\s+(?P<action>.+?)(?:\s+by\s+(?P<due>.+))?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?P<action>.+?)\s+[—-]\s*(?:assigned to|owner[:\s]+)\s*(?P<owner>[A-Z][\w.'-]+(?:\s+[A-Z][\w.'-]+){0,2})"
        r"(?:\s+by\s+(?P<due>.+))?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?P<action>.+?)\s+\((?P<owner>[A-Z][\w.'-]+(?:\s+[A-Z][\w.'-]+){0,2})\)"
        r"(?:\s*[—,-]\s*(?P<due>.+))?$",
    ),
]

_DUE_PATTERNS = [
    re.compile(r"\bby\s+(?P<due>\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})\b", re.I),
    re.compile(r"\bdue\s*:?\s*(?P<due>\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2}|Friday|Monday|next week)\b", re.I),
    re.compile(r"\bbefore\s+(?P<due>\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{1,2})\b", re.I),
]


def _clean_bullet(line: str) -> str:
    text = (line or "").strip()
    text = re.sub(r"^[\s•\-\*\d\.]+", "", text).strip()
    return text


def _extract_action_section(summary: str) -> list[str]:
    lines = (summary or "").splitlines()
    in_actions = False
    out: list[str] = []
    for raw in lines:
        line = raw.strip()
        header = line.rstrip(":").strip()
        if header.lower() in {"action items", "actions", "action item"}:
            in_actions = True
            continue
        if in_actions and header.lower() in {"discussion", "decisions", "decision"}:
            break
        if in_actions and line:
            cleaned = _clean_bullet(line)
            if cleaned:
                out.append(cleaned)
    return out


def extract_action_items(summary: str) -> list[dict[str, Any]]:
    """Parse Action Items prose into ``{owner, action, due_date, text}`` rows."""
    items: list[dict[str, Any]] = []
    for text in _extract_action_section(summary):
        owner = None
        action = text
        due = None
        for pat in _OWNER_PATTERNS:
            m = pat.match(text)
            if m:
                owner = (m.groupdict().get("owner") or "").strip() or None
                action = (m.groupdict().get("action") or text).strip()
                due = (m.groupdict().get("due") or "").strip() or None
                break
        if not due:
            for pat in _DUE_PATTERNS:
                m = pat.search(text)
                if m:
                    due = (m.group("due") or "").strip() or None
                    break
        items.append(
            {
                "text": text,
                "owner": owner,
                "action": action,
                "due_date": due,
                "extracted_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }
        )
    return items
