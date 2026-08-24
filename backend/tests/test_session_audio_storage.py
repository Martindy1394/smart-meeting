"""Tests for disk-primary PCM + Redis rolling-buffer sizing helpers."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import settings  # noqa: E402
from app.services import audio  # noqa: E402
from app.services.redis_store import rolling_buffer_max_bytes  # noqa: E402


def test_rolling_buffer_is_small_vs_hour_of_audio():
    max_bytes = rolling_buffer_max_bytes()
    hour_bytes = settings.audio_sample_rate * 2 * 3600
    assert max_bytes < hour_bytes / 10  # well under 6 minutes by default
    assert max_bytes >= settings.audio_sample_rate * 2 * 30  # at least 30s


def test_disk_pcm_append_and_slice():
    prev = settings.audio_storage_dir
    with tempfile.TemporaryDirectory() as tmp:
        settings.audio_storage_dir = tmp
        mid = "test-meeting-pcm"
        chunk_a = b"\x01\x00" * 100
        chunk_b = b"\x02\x00" * 50
        total = audio.append_raw_pcm(mid, chunk_a)
        assert total == len(chunk_a)
        total = audio.append_raw_pcm(mid, chunk_b)
        assert total == len(chunk_a) + len(chunk_b)
        assert audio.get_raw_pcm_slice(mid, 0, 1) == b"\x01\x00"
        assert audio.read_raw_pcm(mid) == chunk_a + chunk_b
        audio.delete_raw_pcm(mid)
        assert audio.get_raw_pcm_length(mid) == 0
        assert not os.path.exists(audio.raw_pcm_path(mid))
    settings.audio_storage_dir = prev


def test_align_pcm16_drops_odd_trailing_byte():
    assert audio.align_pcm16(b"\x01\x00\x02") == b"\x01\x00"
    assert len(audio.align_pcm16(b"\x01\x00\x02\x00")) == 4


def test_window_hop_overlap_math():
    """10s window / 5s hop must retain 5s overlap (never discard the shared half)."""
    sr = settings.audio_sample_rate
    window = int(10.0 * sr * 2)
    hop = int(5.0 * sr * 2)
    assert window == 320_000
    assert hop == 160_000
    # offsets: 0, hop, 2*hop… each window overlaps previous by window-hop
    assert window - hop == hop


if __name__ == "__main__":
    test_rolling_buffer_is_small_vs_hour_of_audio()
    test_disk_pcm_append_and_slice()
    test_align_pcm16_drops_odd_trailing_byte()
    test_window_hop_overlap_math()
    print("all_session_audio_tests_passed")
