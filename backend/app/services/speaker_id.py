"""Introduction-aware speaker identification on top of Voice N clusters.

This is **not** neural speaker recognition. Whisper transcribes; Voice clusters
group talkers; this module binds a cluster to a roster name when someone
introduces themselves (EN / Tagalog / Hiligaynon) or when an operator correction
is on file.

Optional ``SPEAKER_ID_BACKEND=pyannote`` can overlay diarization timestamps when
pyannote.audio + a Hugging Face token are installed. Default backend is
``heuristic`` so CPU thesis machines keep working.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Iterable

logger = logging.getLogger("smart_meeting.speaker_id")

# Display the roster name on the chip at or above this score.
DISPLAY_THRESHOLD = 0.75
# Escalate / keep Voice N but record a candidate.
CANDIDATE_THRESHOLD = 0.50
# Auto-accept an introduction match against the attendance list.
ROSTER_MATCH_THRESHOLD = 0.82

_NAME = r"([A-Za-zÑñ][\w.'\-]+(?:\s+[A-Za-zÑñ][\w.'\-]+){0,4})"
_INTRO_PATTERNS = [
    re.compile(rf"\bmy\s+name\s+is\s+{_NAME}", re.I),
    re.compile(rf"\bi(?:\s+am|'m)\s+{_NAME}", re.I),
    re.compile(rf"\bthis\s+is\s+{_NAME}", re.I),
    re.compile(rf"\bako(?:\s+po)?\s+si\s+{_NAME}", re.I),
    re.compile(rf"\bang\s+(?:pangalan|ngalan)\s+ko(?:\s+ay)?\s+{_NAME}", re.I),
    re.compile(rf"\bako\s+si\s+{_NAME}", re.I),
]
_STOP = {
    "from",
    "and",
    "kag",
    "ng",
    "sa",
    "of",
    "the",
    "po",
    "gid",
    "here",
    "today",
    "representing",
    "from",
}
_TITLES = re.compile(
    r"^(atty|attorney|dr|dra|hon|engr|eng|rev|fr|sir|maam|ma'am)\.?\s+",
    re.I,
)


@dataclass
class IdentityHit:
    name: str
    confidence: float
    method: str
    guest: bool = False
    extracted: str = ""


@dataclass
class AttendancePerson:
    name: str
    status: str  # present | absent | guest
    first_start: float | None = None
    last_end: float | None = None
    confidence: float = 0.0
    method: str = ""
    speaker_index: int = 0


@dataclass
class AttendanceReport:
    expected: list[str] = field(default_factory=list)
    present: list[AttendancePerson] = field(default_factory=list)
    absent: list[str] = field(default_factory=list)
    guests: list[AttendancePerson] = field(default_factory=list)
    uncertain: int = 0
    backend: str = "heuristic"

    def as_dict(self) -> dict[str, Any]:
        def pack(p: AttendancePerson) -> dict[str, Any]:
            return {
                "name": p.name,
                "status": p.status,
                "first_start": p.first_start,
                "last_end": p.last_end,
                "confidence": round(float(p.confidence or 0), 3),
                "method": p.method,
                "speaker_index": p.speaker_index,
            }

        return {
            "expected": list(self.expected),
            "present": [pack(p) for p in self.present],
            "absent": list(self.absent),
            "guests": [pack(p) for p in self.guests],
            "uncertain": int(self.uncertain),
            "backend": self.backend,
        }


def _split_names(raw) -> list[str]:
    if raw is None:
        return []
    values: list = []
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                import json

                parsed = json.loads(text)
                if isinstance(parsed, list):
                    values = parsed
            except (json.JSONDecodeError, TypeError):
                values = []
        if not values:
            values = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
    elif isinstance(raw, (list, tuple)):
        values = list(raw)
    else:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        name = str(item or "").strip()
        if not name:
            continue
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(name)
    return out


def roster_names(meeting=None, *, attendees=None, presiding_officer=None) -> list[str]:
    """Deduped attendance list: officer + attendees (officer not double-counted)."""
    raw_att = attendees
    raw_off = presiding_officer
    if meeting is not None:
        if raw_att is None:
            raw_att = getattr(meeting, "attendees", None)
        if raw_off is None:
            raw_off = getattr(meeting, "presiding_officer", None)
    names = _split_names(raw_att)
    officer = str(raw_off or "").strip() if isinstance(raw_off, str) else ""
    out: list[str] = []
    seen: set[str] = set()
    for n in ([officer] if officer else []) + names:
        if not n:
            continue
        key = n.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(n)
    return out


def extract_introduction(text: str) -> str | None:
    """Return a spoken self-name if the utterance looks like an introduction."""
    raw = (text or "").strip()
    if not raw:
        return None
    for pat in _INTRO_PATTERNS:
        m = pat.search(raw)
        if not m:
            continue
        name = _clean_extracted_name(m.group(1))
        if name:
            return name
    return None


def _clean_extracted_name(raw: str) -> str | None:
    body = _TITLES.sub("", (raw or "").strip(" .,;:"))
    parts: list[str] = []
    for word in body.split():
        token = word.strip(" .,;:")
        if token.lower() in _STOP and parts:
            break
        if not re.match(r"^[A-Za-zÑñ]", token):
            break
        parts.append(token)
    cleaned = " ".join(parts).strip()
    if len(cleaned) < 2:
        return None
    return cleaned


def match_roster(extracted: str, roster: Iterable[str]) -> tuple[str, float, bool]:
    """Map an extracted name onto the attendance list.

    Returns ``(canonical_name, score, is_guest)``.
    """
    query = (extracted or "").strip()
    names = [n for n in (roster or []) if str(n).strip()]
    if not query:
        return "", 0.0, False
    q = query.casefold()
    best = ("", 0.0)
    for name in names:
        n = str(name).strip()
        cf = n.casefold()
        if cf == q:
            return n, 1.0, False
        if q in cf or cf in q:
            score = 0.92 if min(len(q), len(cf)) >= 4 else 0.84
            if score > best[1]:
                best = (n, score)
            continue
        q_last = q.split()[-1]
        n_last = cf.split()[-1]
        if len(q_last) > 2 and q_last == n_last:
            if 0.86 > best[1]:
                best = (n, 0.86)
            continue
        ratio = SequenceMatcher(None, q, cf).ratio()
        if ratio > best[1]:
            best = (n, ratio)
    if best[1] >= ROSTER_MATCH_THRESHOLD:
        return best[0], float(best[1]), False
    return query, max(0.55, float(best[1] or 0.55)), True


def _seg_text(seg) -> str:
    if isinstance(seg, dict):
        return (seg.get("text") or "").strip()
    return (getattr(seg, "text", None) or "").strip()


def _seg_index(seg) -> int:
    if isinstance(seg, dict):
        try:
            return int(seg.get("speaker_index") or 0)
        except (TypeError, ValueError):
            return 0
    try:
        return int(getattr(seg, "speaker_index", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _seg_times(seg) -> tuple[float, float]:
    from .segment_times import coerce_times

    return coerce_times(seg)


def _set_identity(seg, hit: IdentityHit) -> None:
    if isinstance(seg, dict):
        seg["speaker_name"] = hit.name
        seg["speaker_confidence"] = float(hit.confidence)
        seg["speaker_id_method"] = hit.method
        return
    try:
        seg.speaker_name = hit.name
        seg.speaker_confidence = float(hit.confidence)
        seg.speaker_id_method = hit.method
    except Exception:
        pass


def _asr_penalty(seg) -> float:
    low = False
    if isinstance(seg, dict):
        low = bool(seg.get("low_confidence"))
    else:
        low = bool(getattr(seg, "low_confidence", False))
    return 0.7 if low else 1.0


def identify_segments(
    segments: list,
    meeting=None,
    *,
    attendees=None,
    presiding_officer=None,
    owner_id: str | None = None,
    backend: str | None = None,
) -> list:
    """Annotate Voice-labeled segments with roster names when introductions match."""
    roster = roster_names(
        meeting, attendees=attendees, presiding_officer=presiding_officer
    )
    aliases = {}
    if owner_id:
        try:
            from . import speaker_memory

            aliases = speaker_memory.load_aliases(owner_id)
        except Exception:
            logger.exception("speaker_memory load failed")
            aliases = {}

    cluster_hit: dict[int, IdentityHit] = {}
    for seg in segments or []:
        text = _seg_text(seg)
        idx = _seg_index(seg) or 1
        extracted = extract_introduction(text)
        if not extracted:
            continue
        alias = aliases.get(extracted.casefold())
        if alias:
            hit = IdentityHit(alias, 0.93, "correction", False, extracted)
        else:
            name, score, guest = match_roster(extracted, roster)
            method = "introduction_guest" if guest else "introduction"
            conf = min(0.96, float(score) * _asr_penalty(seg))
            if guest:
                conf = min(conf, 0.68)
            hit = IdentityHit(name, conf, method, guest, extracted)
        prev = cluster_hit.get(idx)
        if prev is None or hit.confidence >= prev.confidence:
            cluster_hit[idx] = hit

    for seg in segments or []:
        idx = _seg_index(seg) or 1
        hit = cluster_hit.get(idx)
        if hit is None:
            continue
        _set_identity(seg, hit)

    used_backend = (backend or _configured_backend()).strip().lower() or "heuristic"
    if used_backend == "pyannote":
        try:
            overlay_pyannote(segments, meeting)
        except Exception:
            logger.exception("pyannote overlay failed; keeping heuristic identities")
    return segments


def overlay_pyannote(segments: list, meeting=None) -> None:
    """Optional pyannote.audio diarization overlay (no-op if not installed)."""
    import os

    path = getattr(meeting, "audio_path", None) or ""
    if not path or not os.path.isfile(path):
        return
    try:
        from pyannote.audio import Pipeline  # type: ignore
    except Exception:
        logger.warning("SPEAKER_ID_BACKEND=pyannote but pyannote.audio is not installed")
        return
    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN") or ""
    if not token:
        logger.warning("pyannote.audio needs HUGGINGFACE_TOKEN for gated models")
        return
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=token
    )
    diarization = pipeline(path)
    # Align overlapping diarization turns onto existing segments by midpoint.
    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append((float(turn.start), float(turn.end), str(speaker)))
    if not turns:
        return
    for seg in segments or []:
        start, end = _seg_times(seg)
        mid = (start + end) / 2.0
        label = ""
        for a, b, spk in turns:
            if a <= mid <= b:
                label = spk
                break
        if not label:
            continue
        if isinstance(seg, dict):
            seg["diarization_label"] = label
        else:
            try:
                seg.diarization_label = label
            except Exception:
                pass


def _configured_backend() -> str:
    try:
        from ..config import settings

        return str(getattr(settings, "speaker_id_backend", "heuristic") or "heuristic")
    except Exception:
        return "heuristic"


def build_attendance_report(
    segments: list,
    meeting=None,
    *,
    attendees=None,
    presiding_officer=None,
) -> AttendanceReport:
    roster = roster_names(
        meeting, attendees=attendees, presiding_officer=presiding_officer
    )
    by_name: dict[str, AttendancePerson] = {}
    uncertain = 0
    for seg in segments or []:
        if isinstance(seg, dict):
            name = (seg.get("speaker_name") or "").strip()
            conf = float(seg.get("speaker_confidence") or 0)
            method = (seg.get("speaker_id_method") or "").strip()
            guest = "guest" in method
        else:
            name = (getattr(seg, "speaker_name", None) or "").strip()
            conf = float(getattr(seg, "speaker_confidence", 0) or 0)
            method = (getattr(seg, "speaker_id_method", None) or "").strip()
            guest = "guest" in method
        start, end = _seg_times(seg)
        idx = _seg_index(seg)
        if not name or conf < CANDIDATE_THRESHOLD:
            if _seg_text(seg):
                uncertain += 1
            continue
        key = name.casefold()
        row = by_name.get(key)
        if row is None:
            status = "guest" if guest and all(name.casefold() != r.casefold() for r in roster) else "present"
            row = AttendancePerson(
                name=name,
                status=status,
                first_start=start,
                last_end=end,
                confidence=conf,
                method=method,
                speaker_index=idx,
            )
            by_name[key] = row
        else:
            row.confidence = max(row.confidence, conf)
            if row.first_start is None or start < row.first_start:
                row.first_start = start
            if row.last_end is None or end > row.last_end:
                row.last_end = end
            if method and not row.method:
                row.method = method
    present = [p for p in by_name.values() if p.status == "present"]
    guests = [p for p in by_name.values() if p.status == "guest"]
    present_keys = {p.name.casefold() for p in present}
    absent = [n for n in roster if n.casefold() not in present_keys]
    return AttendanceReport(
        expected=list(roster),
        present=sorted(present, key=lambda p: (p.first_start is None, p.first_start or 0)),
        absent=absent,
        guests=guests,
        uncertain=uncertain,
        backend=_configured_backend(),
    )


def format_attendance_text(report: AttendanceReport | dict) -> str:
    if isinstance(report, dict):
        expected = report.get("expected") or []
        present = report.get("present") or []
        guests = report.get("guests") or []
        absent = report.get("absent") or []
        uncertain = int(report.get("uncertain") or 0)
        lines = ["Expected: " + (", ".join(expected) or "—")]
        if present:
            lines.append("Present:")
            for p in present:
                name = p.get("name") or ""
                ts = ""
                if p.get("first_start") is not None:
                    ts = f" @ {float(p['first_start']):.1f}s"
                conf = float(p.get("confidence") or 0)
                method = p.get("method") or "unknown"
                lines.append(f"  - {name}{ts} (conf {conf:.2f}, {method})")
        else:
            lines.append("Present: —")
        if guests:
            lines.append("Guests (temporary enrollment from introductions):")
            for p in guests:
                lines.append(
                    f"  - {p.get('name')} (conf {float(p.get('confidence') or 0):.2f})"
                )
        if absent:
            lines.append("Absent: " + ", ".join(absent))
        if uncertain:
            lines.append(f"Unidentified turns: {uncertain}")
        return "\n".join(lines)
    lines = ["Expected: " + (", ".join(report.expected) or "—")]
    if report.present:
        lines.append("Present:")
        for p in report.present:
            ts = ""
            if p.first_start is not None:
                ts = f" @ {p.first_start:.1f}s"
            lines.append(
                f"  - {p.name}{ts} (conf {p.confidence:.2f}, {p.method or 'unknown'})"
            )
    else:
        lines.append("Present: —")
    if report.guests:
        lines.append("Guests (temporary enrollment from introductions):")
        for p in report.guests:
            lines.append(f"  - {p.name} (conf {p.confidence:.2f})")
    if report.absent:
        lines.append("Absent: " + ", ".join(report.absent))
    if report.uncertain:
        lines.append(f"Unidentified turns: {report.uncertain}")
    return "\n".join(lines)
