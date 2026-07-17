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
    merge_live_caption,
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


if __name__ == "__main__":
    test_merge_live_caption_appends_novel_overlap()
    test_merge_live_caption_never_shrinks()
    test_model_cache_lru_eviction()
    test_per_model_locks_are_distinct()
    print("all_unit_tests_passed")
