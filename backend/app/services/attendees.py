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


def _meeting_when(meeting: Any):
    if isinstance(meeting, dict):
        return meeting.get("meeting_date") or meeting.get("created_at")
    return getattr(meeting, "meeting_date", None) or getattr(meeting, "created_at", None)


def _meeting_title(meeting: Any) -> str:
    if isinstance(meeting, dict):
        return str(meeting.get("title") or "").strip()
    return str(getattr(meeting, "title", None) or "").strip()


def _meeting_venue(meeting: Any) -> str:
    if isinstance(meeting, dict):
        return str(meeting.get("venue") or "").strip()
    return str(getattr(meeting, "venue", None) or "").strip()


def _identified_names(meeting: Any) -> list[tuple[str, float]]:
    raw = None
    if isinstance(meeting, dict):
        raw = meeting.get("speaker_attendance_json") or meeting.get("attendance")
    else:
        raw = getattr(meeting, "speaker_attendance_json", None)
    payload = raw
    if isinstance(raw, str) and raw.strip():
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            payload = None
    if not isinstance(payload, dict):
        return []
    out: list[tuple[str, float]] = []
    for bucket in ("present", "guests"):
        for row in payload.get(bucket) or []:
            if isinstance(row, str):
                name = normalize_attendee_name(row)
                conf = 0.0
            elif isinstance(row, dict):
                name = normalize_attendee_name(row.get("name"))
                try:
                    conf = float(row.get("confidence") or 0)
                except (TypeError, ValueError):
                    conf = 0.0
            else:
                continue
            if name:
                out.append((name, conf))
    return out


def collect_name_directory(
    meetings: Iterable[Any],
    *,
    max_officers: int = 80,
    max_attendees: int = 120,
) -> dict[str, Any]:
    """Unique presiding officers and attendees, newest meetings first.

    Also returns ``people`` rows with frequency, last meeting, and whether the
    name was confirmed by speech identification.
    """
    people: dict[str, dict[str, Any]] = {}

    def bump(
        name: str,
        *,
        as_officer: bool = False,
        as_attendee: bool = False,
        identified: bool = False,
        when=None,
        title: str = "",
    ) -> None:
        key = name.casefold()
        row = people.get(key)
        if row is None:
            row = {
                "name": name,
                "officer_count": 0,
                "attendee_count": 0,
                "identified_count": 0,
                "last_seen": when,
                "last_title": title,
                "sources": [],
            }
            people[key] = row
        if as_officer:
            row["officer_count"] += 1
            if "officer" not in row["sources"]:
                row["sources"].append("officer")
        if as_attendee:
            row["attendee_count"] += 1
            if "attendee" not in row["sources"]:
                row["sources"].append("attendee")
        if identified:
            row["identified_count"] += 1
            if "identified" not in row["sources"]:
                row["sources"].append("identified")
        if when and not row["last_seen"]:
            row["last_seen"] = when
        if title and not row["last_title"]:
            row["last_title"] = title

    officer_order: list[str] = []
    attendee_order: list[str] = []
    title_order: list[str] = []
    venue_order: list[str] = []
    seen_off: set[str] = set()
    seen_att: set[str] = set()
    seen_titles: set[str] = set()
    seen_venues: set[str] = set()
    skip_titles = {"untitled meeting", "untitled"}

    for meeting in meetings or []:
        when = _meeting_when(meeting)
        title = _meeting_title(meeting)
        venue = _meeting_venue(meeting)
        if title and title.casefold() not in skip_titles:
            tkey = title.casefold()
            if tkey not in seen_titles:
                seen_titles.add(tkey)
                title_order.append(title)
        if venue:
            vkey = venue.casefold()
            if vkey not in seen_venues:
                seen_venues.add(vkey)
                venue_order.append(venue)
        officer = getattr(meeting, "presiding_officer", None)
        if isinstance(meeting, dict):
            officer = meeting.get("presiding_officer", officer)
        name = normalize_attendee_name(officer) if isinstance(officer, str) else None
        if name:
            bump(name, as_officer=True, when=when, title=title)
            key = name.casefold()
            if key not in seen_off:
                seen_off.add(key)
                officer_order.append(name)
        raw_att = getattr(meeting, "attendees", None)
        if isinstance(meeting, dict):
            raw_att = meeting.get("attendees", raw_att)
        for attendee in load_attendees(raw_att):
            bump(attendee, as_attendee=True, when=when, title=title)
            key = attendee.casefold()
            if key not in seen_att:
                seen_att.add(key)
                attendee_order.append(attendee)
        for ident, _conf in _identified_names(meeting):
            bump(ident, identified=True, when=when, title=title)
            key = ident.casefold()
            if key not in seen_att:
                seen_att.add(key)
                attendee_order.append(ident)

    ranked_people = sorted(
        people.values(),
        key=lambda r: (
            -(int(r["officer_count"]) + int(r["attendee_count"])),
            -(int(r["identified_count"])),
        ),
    )
    return {
        "presiding_officers": officer_order[:max_officers],
        "attendees": attendee_order[:max_attendees],
        "people": ranked_people[: max(max_officers, max_attendees)],
        "titles": title_order[:80],
        "venues": venue_order[:80],
    }


class AttendeesJSON(TypeDecorator):
    """SQLAlchemy column type: Python ``list[str]`` ↔ JSON text in the DB."""

    impl = Text
    cache_ok = True

    def process_bind_param(self, value: Any, dialect) -> str:
        return dump_attendees(value)

    def process_result_value(self, value: Any, dialect) -> list[str]:
        return load_attendees(value)
