#!/usr/bin/env python3
"""Print the production full-file Whisper kwargs vs the manual WAV repro."""
from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.transcription import (  # noqa: E402
    LanguageDetection,
    _FINAL_VAD_PARAMS,
    _faster_whisper_final_kwargs,
    _final_decode_prompt,
)

# Copied from the one-shot /tmp English decode (VAD off, ad-hoc prompt).
MANUAL_REPRO = {
    "language": "en",
    "task": "transcribe",
    "beam_size": 5,
    "best_of": 5,
    "temperature": [0.0, 0.2],
    "vad_filter": False,
    "condition_on_previous_text": True,
    "without_timestamps": False,
    "initial_prompt": "English board meeting. Mic test. Names of attendees.",
    "no_speech_threshold": 0.25,
    "compression_ratio_threshold": 2.6,
    "log_prob_threshold": -1.2,
}


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def main() -> None:
    high = LanguageDetection(language="en", confidence=0.88, detected_by="whisper")
    prompt = _final_decode_prompt("auto", extra_terms=None, detection=high)
    production = _faster_whisper_final_kwargs(
        language="en",
        vad_filter=True,
        initial_prompt=prompt,
    )
    keys = sorted(set(MANUAL_REPRO) | set(production))
    deltas = {}
    for key in keys:
        old = MANUAL_REPRO.get(key, "<missing>")
        new = production.get(key, "<missing>")
        if old != new:
            deltas[key] = {"repro": old, "production": new}
    payload = {
        "repro": MANUAL_REPRO,
        "production": _jsonable(production),
        "deltas": _jsonable(deltas),
        "vad_parameters": _FINAL_VAD_PARAMS,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
