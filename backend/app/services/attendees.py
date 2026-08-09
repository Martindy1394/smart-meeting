"""Attendee list bridging: API ``list[str]`` ↔ DB JSON ``Text``.

Single source of truth for clean/serialize/parse so routers, schemas, and the
ORM TypeDecorator cannot drift.
"""
from __future__ import annotations

import json
from typing import Any, Iterable

from sqlalchemy import Text, TypeDecorator


def normalize_attendee_name(name: Any) -> str | None:
    """Return a stripped non-empty name, or ``None`` if unusable."""
    if not isinstance(name, str):
        return None
    cleaned = name.strip()
    return cleaned or None


def normalize_attendees(names: Iterable[Any] | None) -> list[str]:
    """Type-safe clean + de-dupe (order-preserving) for API/ORM use.

    Dedupes case-insensitively while keeping the first-seen spelling.
    """
    if names is None:
        return []
    if isinstance(names, str):
        # Accidental raw JSON string — parse then clean.
        names = load_attendees(names)
    seen: set[str] = set()
    out: list[str] = []
    for raw in names:
        name = normalize_attendee_name(raw)
        if name is None:
            continue
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(name)
    return out


def dump_attendees(names: Iterable[Any] | None) -> str:
    """Serialize attendees to the DB JSON-text representation."""
    return json.dumps(normalize_attendees(names), ensure_ascii=False)


def load_attendees(raw: Any) -> list[str]:
    """Parse DB/API payload into a clean ``list[str]``.

    Accepts JSON text, a list, ``None``, or garbage — never raises.
    """
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        return normalize_attendees(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            # Legacy plain comma-separated names.
            if "," in raw:
                return normalize_attendees(raw.split(","))
            single = normalize_attendee_name(raw)
            return [single] if single else []
        if isinstance(parsed, list):
            return normalize_attendees(parsed)
        return []
    return []


class AttendeesJSON(TypeDecorator):
    """SQLAlchemy column type: Python ``list[str]`` ↔ JSON text in the DB."""

    impl = Text
    cache_ok = True

    def process_bind_param(self, value: Any, dialect) -> str:
        return dump_attendees(value)

    def process_result_value(self, value: Any, dialect) -> list[str]:
        return load_attendees(value)
