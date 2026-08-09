"""Process-wide live-transcription metrics for health / backpressure."""
from __future__ import annotations

import threading

_lock = threading.Lock()
_active_sessions = 0
_backlog_windows = 0
_windows_dropped = 0
_last_finalize_ok: bool | None = None


def session_started() -> None:
    global _active_sessions
    with _lock:
        _active_sessions += 1


def session_ended() -> None:
    global _active_sessions
    with _lock:
        _active_sessions = max(0, _active_sessions - 1)


def set_backlog_windows(n: int) -> None:
    global _backlog_windows
    with _lock:
        _backlog_windows = max(0, int(n))


def record_windows_dropped(n: int) -> None:
    global _windows_dropped
    if n <= 0:
        return
    with _lock:
        _windows_dropped += int(n)


def record_finalize(ok: bool) -> None:
    global _last_finalize_ok
    with _lock:
        _last_finalize_ok = bool(ok)


def snapshot() -> dict:
    with _lock:
        return {
            "active_live_sessions": _active_sessions,
            "live_asr_backlog_windows": _backlog_windows,
            "live_asr_windows_dropped_total": _windows_dropped,
            "last_finalize_ok": _last_finalize_ok,
        }
