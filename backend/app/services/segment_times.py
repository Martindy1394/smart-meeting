"""Segment timestamp bridging: wire ``start``/``end`` ↔ DB/API ``start_time``/``end_time``.

Whisper/finalize/WS historically emit ``start``/``end``. The ORM and REST detail
schema use ``start_time``/``end_time``. This module is the single mapping layer.
"""
from __future__ import annotations

from typing import Any, Mapping


def coerce_times(payload: Mapping[str, Any] | Any) -> tuple[float, float]:
    """Extract ``(start_time, end_time)`` from either key pair.

    Prefer explicit ``start_time``/``end_time``; fall back to ``start``/``end``.
    """
    if payload is None:
        return 0.0, 0.0

    def _get(primary: str, alias: str) -> float:
        if hasattr(payload, primary) or hasattr(payload, alias):
            raw = getattr(payload, primary, None)
            if raw is None:
                raw = getattr(payload, alias, None)
        elif isinstance(payload, Mapping):
            raw = payload.get(primary, payload.get(alias))
        else:
            raw = None
        try:
            return float(raw if raw is not None else 0.0)
        except (TypeError, ValueError):
            return 0.0

    start = _get("start_time", "start")
    end = _get("end_time", "end")
    if end < start:
        end = start
    return start, end


def absolute_window_times(
    *,
    byte_offset: int | None,
    sample_rate: int,
    relative_start: float,
    relative_end: float,
    window_duration: float = 0.0,
) -> tuple[float, float]:
    """Map in-window Whisper times onto absolute meeting timeline seconds.

    Live ASR segments are relative to the PCM window; ``byte_offset`` is the
    window start in the meeting PCM stream (int16 mono → 2 bytes/sample).
    """
    rate = max(1, int(sample_rate))
    base = float(byte_offset or 0) / float(rate * 2)
    start = base + max(0.0, float(relative_start or 0.0))
    end = base + max(0.0, float(relative_end or 0.0))
    if end <= start and window_duration > 0:
        end = base + float(window_duration)
    if end < start:
        end = start
    return start, end


def segment_wire_dict(
    *,
    text: str,
    start: float,
    end: float,
    seq: int | None = None,
    kind: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Canonical segment payload with both wire and API timestamp keys."""
    start_f = float(start or 0.0)
    end_f = float(end or 0.0)
    if end_f < start_f:
        end_f = start_f
    out: dict[str, Any] = {
        "text": text or "",
        # Wire / finalize / WS (legacy + current clients)
        "start": start_f,
        "end": end_f,
        # DB / REST detail alignment
        "start_time": start_f,
        "end_time": end_f,
    }
    if seq is not None:
        out["seq"] = int(seq)
    if kind is not None:
        out["kind"] = kind
    out.update(extra)
    return out


def segments_from_asr(segments: list[Any]) -> list[dict[str, Any]]:
    """Serialize ASR ``Segment`` objects (``.start``/``.end``) to wire dicts."""
    out: list[dict[str, Any]] = []
    for i, seg in enumerate(segments or []):
        start, end = coerce_times(seg)
        text = getattr(seg, "text", None)
        if text is None and isinstance(seg, Mapping):
            text = seg.get("text", "")
        out.append(
            segment_wire_dict(
                text=str(text or ""),
                start=start,
                end=end,
                seq=i,
            )
        )
    return out
