"""Unit tests for transcription reliability helpers (no Whisper weights required)."""
from __future__ import annotations

import sys
import threading
from pathlib import Path

# Allow `python -m pytest` / direct run from repo root or backend/.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.transcription import (  # noqa: E402
    _ModelCache,
    _forced_language,
    final_faster_model_id,
    hiligaynon_hf_candidates,
    hiligaynon_model_id,
    is_philippine_language,
    live_model_id,
    merge_live_caption,
    resolve_final_backend,
    set_model_cache,
)


def test_merge_live_caption_appends_novel_overlap():
    prev = "good morning everyone welcome to the board meeting"
    prev_window = "welcome to the board meeting today we will"
    cur_window = "board meeting today we will discuss the budget"
    merged = merge_live_caption(prev, cur_window, previous_window=prev_window)
    assert merged.startswith(prev)
    assert "discuss the budget" in merged
    assert len(merged.split()) >= len(prev.split())


def test_merge_live_caption_never_shrinks():
    prev = "one two three four five six seven eight"
    merged = merge_live_caption(prev, "one two three", previous_window=prev)
    assert len(merged.split()) >= len(prev.split())


def test_model_cache_lru_eviction():
    cache = _ModelCache(max_models=2)
    set_model_cache(cache)
    try:
        with cache._lock:
            cache._fw_models["a"] = object()
            cache._fw_locks["a"] = threading.Lock()
            cache._touch("fw", "a")
            cache._fw_models["b"] = object()
            cache._fw_locks["b"] = threading.Lock()
            cache._touch("fw", "b")
            # Touch a so b becomes the LRU victim when c is inserted.
            cache._touch("fw", "a")
            cache._fw_models["c"] = object()
            cache._fw_locks["c"] = threading.Lock()
            cache._touch("fw", "c")
            assert "b" not in cache._fw_models
            assert "a" in cache._fw_models
            assert "c" in cache._fw_models
            assert cache.stats()["max_models"] == 2
    finally:
        set_model_cache(None)


def test_per_model_locks_are_distinct():
    cache = _ModelCache(max_models=2)
    lock_a = cache.fw_infer_lock("small")
    lock_b = cache.fw_infer_lock("medium")
    assert lock_a is not lock_b
    assert cache.fw_infer_lock("small") is lock_a


def test_hiligaynon_maps_to_whisper_tl():
    assert is_philippine_language("hil")
    assert is_philippine_language("Hiligaynon")
    assert is_philippine_language("tl")
    assert not is_philippine_language("en")
    # Whisper has no native hil token — decode uses configured PH code.
    assert _forced_language("hil") == "tl"
    assert _forced_language("fil") == "tl"
    assert _forced_language("en") == "en"


def test_auto_final_backend_prefers_hf_for_hiligaynon(monkeypatch=None):
    # Prefer HF fine-tune for hil when backend=auto.
    from app.config import settings

    prev = settings.whisper_final_backend
    prev_live = settings.whisper_live_hiligaynon_model
    prev_ft = settings.whisper_hiligaynon_fine_tuned_model
    prev_ph = settings.whisper_hiligaynon_model
    try:
        settings.whisper_final_backend = "auto"
        settings.whisper_hiligaynon_fine_tuned_model = ""
        settings.whisper_hiligaynon_model = "rbcurzon/whisper-medium-ph"
        assert resolve_final_backend("hil") == "huggingface"
        assert resolve_final_backend("en") == "faster-whisper"
        assert hiligaynon_hf_candidates() == ["rbcurzon/whisper-medium-ph"]
        assert hiligaynon_model_id() == "rbcurzon/whisper-medium-ph"

        settings.whisper_hiligaynon_fine_tuned_model = "/models/my-hil-ft"
        assert hiligaynon_hf_candidates() == [
            "/models/my-hil-ft",
            "rbcurzon/whisper-medium-ph",
        ]
        assert hiligaynon_model_id() == "/models/my-hil-ft"

        settings.whisper_live_hiligaynon_model = "/models/hil-ct2"
        assert live_model_id("hil") == "/models/hil-ct2"
        assert live_model_id("en") == settings.whisper_live_model
        assert final_faster_model_id("hil") == "/models/hil-ct2"
    finally:
        settings.whisper_final_backend = prev
        settings.whisper_live_hiligaynon_model = prev_live
        settings.whisper_hiligaynon_fine_tuned_model = prev_ft
        settings.whisper_hiligaynon_model = prev_ph


if __name__ == "__main__":
    test_merge_live_caption_appends_novel_overlap()
    test_merge_live_caption_never_shrinks()
    test_model_cache_lru_eviction()
    test_per_model_locks_are_distinct()
    test_hiligaynon_maps_to_whisper_tl()
    test_auto_final_backend_prefers_hf_for_hiligaynon()
    print("all_unit_tests_passed")
