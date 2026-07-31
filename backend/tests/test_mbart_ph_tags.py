"""Tagalog/Hiligaynon mBART tag + routing regressions (no full model weights)."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import languages
from app.config import settings
from app.services import llm


def test_stock_tagalog_mbart_code_is_tl_xx():
    with patch.object(settings, "mbart_ph_finetuned_model", ""):
        assert languages.mbart_code("tl") == "tl_XX"
        assert languages.mbart_code("tagalog") == "tl_XX"
        assert languages.mbart_code("fil") == "tl_XX"


def test_hiligaynon_mbart_code_is_degraded_id_proxy():
    with patch.object(settings, "mbart_ph_finetuned_model", ""):
        assert languages.mbart_code("hil") == "id_ID"
        assert languages.mbart_code("hiligaynon") == "id_ID"
        assert languages.LANGUAGES["hil"].get("fallback") is True


def test_ph_finetune_maps_hil_and_tl_to_tl_xx():
    with patch.object(settings, "mbart_ph_finetuned_model", "/models/mbart-ph"):
        assert languages.mbart_code("tl") == "tl_XX"
        assert languages.mbart_code("hil") == "tl_XX"


def test_assert_mbart_codes_resolvable():
    languages.assert_mbart_codes_resolvable()


def test_route_attempts_tagalog_mbart_uses_tl_not_id():
    attempts = llm._route_attempts_for_line(
        "tl", prefer_mbart=False, has_ph_mbart=False, use_nllb=True
    )
    assert ("nllb", "tl") in attempts
    assert ("mbart", "tl") in attempts
    assert ("mbart", "id") not in attempts


def test_route_attempts_hiligaynon_mbart_is_last_resort():
    with patch(
        "app.services.google_translate.is_configured", return_value=True
    ):
        attempts = llm._route_attempts_for_line(
            "hil", prefer_mbart=False, has_ph_mbart=False, use_nllb=True
        )
    assert attempts[0] == ("google", "hil")
    assert ("nllb", "hil") in attempts
    assert attempts[-1] == ("mbart", "id")


def test_unmapped_mbart_source_logs_warning_not_silent_en():
    import torch

    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    class _Tok:
        lang_code_to_id = {"id_ID": 1, "en_XX": 2, "tl_XX": 3}

        def __init__(self):
            self.src_lang = None

        def __call__(self, *a, **k):
            return {
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.tensor([[1, 1]]),
            }

        def convert_tokens_to_ids(self, t):
            return self.lang_code_to_id.get(t, 0)

        def batch_decode(self, ids, skip_special_tokens=True):
            return ["Hello from proxy."]

    class _Model:
        def generate(self, **kwargs):
            return torch.tensor([[1, 2, 3]])

    class _NullLock:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    handler = _Capture()
    log = logging.getLogger("smart_meeting.llm")
    prev = log.level
    log.addHandler(handler)
    log.setLevel(logging.WARNING)
    try:
        with (
            patch.object(llm._pipelines, "mbart", return_value=(_Model(), _Tok())),
            patch.object(llm._pipelines, "mbart_infer_lock", return_value=_NullLock()),
            patch.object(llm, "mbart_code", side_effect=lambda c: "en_XX" if c == "en" else None),
        ):
            out = llm._mbart_translate("test text here", "not_a_real_lang", "en")
    finally:
        log.removeHandler(handler)
        log.setLevel(prev)

    assert out.strip()
    assert any("Unmapped mBART source code" in r.getMessage() for r in records)
    # Must not have silently chosen en_XX as the *source* without warning.
    assert not any("silent" in r.getMessage().lower() and "en_xx" in r.getMessage().lower()
                   for r in records if "Unmapped" not in r.getMessage())


if __name__ == "__main__":
    test_stock_tagalog_mbart_code_is_tl_xx()
    test_hiligaynon_mbart_code_is_degraded_id_proxy()
    test_ph_finetune_maps_hil_and_tl_to_tl_xx()
    test_assert_mbart_codes_resolvable()
    test_route_attempts_tagalog_mbart_uses_tl_not_id()
    test_route_attempts_hiligaynon_mbart_is_last_resort()
    test_unmapped_mbart_source_logs_warning_not_silent_en()
    print("all_mbart_ph_tag_tests_passed")
