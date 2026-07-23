"""Unit tests for optional NeMo RNN-T live backend selection (no NeMo required)."""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import settings  # noqa: E402
from app.services import rnnt  # noqa: E402


def test_should_use_rnnt_for_ph_when_auto():
    prev = settings.whisper_live_backend
    try:
        settings.whisper_live_backend = "auto"
        assert rnnt.should_use_rnnt_live("auto") is True
        assert rnnt.should_use_rnnt_live("hil") is True
        assert rnnt.should_use_rnnt_live("tl") is True
        assert rnnt.should_use_rnnt_live("en") is False
    finally:
        settings.whisper_live_backend = prev


def test_whisper_live_backend_disables_rnnt():
    prev = settings.whisper_live_backend
    try:
        settings.whisper_live_backend = "whisper"
        assert rnnt.should_use_rnnt_live("hil") is False
        assert rnnt.should_use_rnnt_live("auto") is False
    finally:
        settings.whisper_live_backend = prev


def test_rnnt_status_shape():
    st = rnnt.status()
    assert "configured_backend" in st
    assert "model" in st
    assert "nemo_importable" in st
    assert st["model"] == rnnt.rnnt_model_id()
