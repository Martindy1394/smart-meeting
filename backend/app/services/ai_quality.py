"""Persist AI quality flags (extractive fallback + faithfulness) on meetings."""
from __future__ import annotations

import json
from typing import Any


def dump_faithfulness(report: Any) -> str:
    """Serialize a faithfulness report dict/model to DB JSON text."""
    if report is None or report == "":
        return ""
    if hasattr(report, "model_dump"):
        report = report.model_dump()
    if isinstance(report, str):
        # Already JSON or empty.
        raw = report.strip()
        if not raw:
            return ""
        try:
            json.loads(raw)
            return raw
        except (json.JSONDecodeError, TypeError):
            return ""
    if isinstance(report, dict):
        return json.dumps(report, ensure_ascii=False)
    return ""


def load_faithfulness(raw: Any) -> dict | None:
    """Parse stored faithfulness JSON into a plain dict (or ``None``)."""
    if raw is None or raw == "":
        return None
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None
