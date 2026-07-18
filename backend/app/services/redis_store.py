"""Redis-backed memory storage for *recent* live audio and session state.

Full multi-hour PCM lives on disk (``audio.append_raw_pcm``). Redis keeps only:
* a rolling PCM window (last ``redis_live_buffer_seconds``) for reconnect assist
* session meta (caption / offset / seq / last_active)
* finalized WAV cache

This avoids the Redis string 512 MiB ceiling and OOM on long board meetings.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from ..config import settings

logger = logging.getLogger("smart_meeting.redis")

_client = None
_client_failed = False


def _pcm_key(meeting_id: str) -> str:
    return f"smartmeeting:audio:{meeting_id}:pcm"


def _wav_key(meeting_id: str) -> str:
    return f"smartmeeting:audio:{meeting_id}:wav"


def _meta_key(meeting_id: str) -> str:
    return f"smartmeeting:audio:{meeting_id}:meta"


def get_client():
    """Return a shared Redis client, or ``None`` if Redis is unavailable."""
    global _client, _client_failed
    if _client is not None:
        return _client
    if _client_failed:
        return None
    try:
        import redis  # type: ignore
    except Exception as exc:  # pragma: no cover
        logger.warning("redis package not installed: %s", exc)
        _client_failed = True
        return None

    try:
        client = redis.Redis.from_url(
            settings.redis_url,
            decode_responses=False,
            socket_connect_timeout=2.0,
            socket_timeout=30.0,
            health_check_interval=30,
        )
        client.ping()
        _client = client
        logger.info("Connected to Redis at %s", settings.redis_url)
        return _client
    except Exception as exc:
        logger.warning("Redis unavailable (%s); falling back to in-process buffers.", exc)
        _client_failed = True
        return None


def is_available() -> bool:
    client = get_client()
    if client is None:
        return False
    try:
        client.ping()
        return True
    except Exception:
        return False


def reset_client_for_tests() -> None:
    """Clear cached client state (tests / reconnect after Redis comes up)."""
    global _client, _client_failed
    _client = None
    _client_failed = False


def rolling_buffer_max_bytes() -> int:
    seconds = max(30.0, float(settings.redis_live_buffer_seconds))
    return int(seconds * settings.audio_sample_rate * 2)


def _touch_ttl(client, meeting_id: str) -> None:
    ttl = int(settings.redis_audio_ttl_seconds)
    if ttl <= 0:
        return
    client.expire(_pcm_key(meeting_id), ttl)
    client.expire(_meta_key(meeting_id), ttl)
    client.expire(_wav_key(meeting_id), ttl)


def append_pcm(meeting_id: str, chunk: bytes) -> int:
    """Append PCM into the Redis *rolling* buffer (not the full recording).

    Returns the rolling buffer byte length after append, or ``-1`` on failure.
    Full meeting audio must be written to disk separately.
    """
    if not chunk:
        return get_pcm_length(meeting_id)
    client = get_client()
    if client is None:
        return -1
    max_bytes = rolling_buffer_max_bytes()
    key = _pcm_key(meeting_id)
    try:
        # Truncate oversized incoming chunks so one write cannot blow the cap.
        if len(chunk) > max_bytes:
            chunk = chunk[-max_bytes:]
        total = int(client.append(key, chunk))
        if total > max_bytes:
            # Keep only the newest window — cheap for ~MB-scale buffers.
            kept = client.getrange(key, total - max_bytes, -1) or b""
            client.set(key, kept)
            total = len(kept)
        _touch_ttl(client, meeting_id)
        return total
    except Exception as exc:
        logger.exception("Redis rolling APPEND failed for meeting %s: %s", meeting_id, exc)
        return -1


def get_pcm(meeting_id: str) -> bytes:
    """Return the Redis rolling PCM window (not the full multi-hour recording)."""
    client = get_client()
    if client is None:
        return b""
    try:
        data = client.get(_pcm_key(meeting_id))
        return data or b""
    except Exception as exc:
        logger.exception("Redis GET pcm failed for meeting %s: %s", meeting_id, exc)
        return b""


def iter_pcm_chunks(meeting_id: str, chunk_bytes: int):
    """Yield successive PCM slices from the Redis rolling buffer."""
    data = get_pcm(meeting_id)
    if not data:
        return
    if chunk_bytes <= 0:
        yield data
        return
    start = 0
    while start < len(data):
        yield data[start : start + chunk_bytes]
        start += chunk_bytes


def get_pcm_length(meeting_id: str) -> int:
    client = get_client()
    if client is None:
        return 0
    try:
        return int(client.strlen(_pcm_key(meeting_id)))
    except Exception:
        return 0


def get_pcm_slice(meeting_id: str, start: int, end: int) -> bytes:
    """Inclusive Redis GETRANGE slice ``[start, end]`` within the rolling buffer."""
    client = get_client()
    if client is None:
        return b""
    if end < start:
        return b""
    try:
        data = client.getrange(_pcm_key(meeting_id), start, end)
        return data or b""
    except Exception as exc:
        logger.exception("Redis GETRANGE failed for meeting %s: %s", meeting_id, exc)
        return b""


def save_wav_bytes(meeting_id: str, wav_bytes: bytes) -> bool:
    """Cache finalized WAV bytes in Redis memory storage."""
    if not wav_bytes:
        return False
    client = get_client()
    if client is None:
        return False
    try:
        client.set(_wav_key(meeting_id), wav_bytes)
        _touch_ttl(client, meeting_id)
        return True
    except Exception as exc:
        logger.exception("Redis SET wav failed for meeting %s: %s", meeting_id, exc)
        return False


def get_wav_bytes(meeting_id: str) -> bytes:
    client = get_client()
    if client is None:
        return b""
    try:
        data = client.get(_wav_key(meeting_id))
        return data or b""
    except Exception:
        return b""


def set_session_meta(
    meeting_id: str,
    *,
    live_caption: Optional[str] = None,
    previous_window: Optional[str] = None,
    live_offset: Optional[int] = None,
    seq: Optional[int] = None,
    last_active: Optional[float] = None,
) -> None:
    """Persist live-session fields so reconnects can resume cleanly."""
    client = get_client()
    if client is None:
        return
    mapping = {}
    if live_caption is not None:
        mapping[b"live_caption"] = live_caption.encode("utf-8")
    if previous_window is not None:
        mapping[b"previous_window"] = previous_window.encode("utf-8")
    if live_offset is not None:
        mapping[b"live_offset"] = str(int(live_offset)).encode("utf-8")
    if seq is not None:
        mapping[b"seq"] = str(int(seq)).encode("utf-8")
    # Always refresh activity when meta is written so the janitor can age sessions.
    ts = time.time() if last_active is None else float(last_active)
    mapping[b"last_active"] = str(ts).encode("utf-8")
    try:
        client.hset(_meta_key(meeting_id), mapping=mapping)
        _touch_ttl(client, meeting_id)
    except Exception as exc:
        logger.exception("Redis HSET meta failed for meeting %s: %s", meeting_id, exc)


def touch_session(meeting_id: str) -> None:
    """Bump last_active without changing caption/offset."""
    set_session_meta(meeting_id, last_active=time.time())


def get_session_meta(meeting_id: str) -> dict:
    client = get_client()
    if client is None:
        return {}
    try:
        raw = client.hgetall(_meta_key(meeting_id)) or {}
    except Exception as exc:
        logger.exception("Redis HGETALL meta failed for meeting %s: %s", meeting_id, exc)
        return {}

    def _s(key: bytes, default: str = "") -> str:
        val = raw.get(key)
        if val is None:
            return default
        return val.decode("utf-8") if isinstance(val, (bytes, bytearray)) else str(val)

    def _i(key: bytes, default: int = 0) -> int:
        try:
            return int(_s(key, str(default)))
        except ValueError:
            return default

    def _f(key: bytes, default: float = 0.0) -> float:
        try:
            return float(_s(key, str(default)))
        except ValueError:
            return default

    return {
        "live_caption": _s(b"live_caption"),
        "previous_window": _s(b"previous_window"),
        "live_offset": _i(b"live_offset"),
        "seq": _i(b"seq"),
        "last_active": _f(b"last_active"),
    }


def list_session_meeting_ids() -> list[str]:
    """Return meeting ids that currently have Redis session meta keys."""
    client = get_client()
    if client is None:
        return []
    prefix = b"smartmeeting:audio:"
    suffix = b":meta"
    out: list[str] = []
    try:
        for key in client.scan_iter(match="smartmeeting:audio:*:meta", count=100):
            raw = key if isinstance(key, (bytes, bytearray)) else str(key).encode()
            if not raw.startswith(prefix) or not raw.endswith(suffix):
                continue
            mid = raw[len(prefix) : -len(suffix)].decode("utf-8", errors="ignore")
            if mid:
                out.append(mid)
    except Exception as exc:
        logger.exception("Redis SCAN sessions failed: %s", exc)
    return out


def clear_meeting_audio(meeting_id: str, *, keep_wav: bool = False) -> None:
    """Remove live PCM/meta (and optionally WAV) from Redis."""
    client = get_client()
    if client is None:
        return
    keys = [_pcm_key(meeting_id), _meta_key(meeting_id)]
    if not keep_wav:
        keys.append(_wav_key(meeting_id))
    try:
        client.delete(*keys)
    except Exception as exc:
        logger.exception("Redis DELETE failed for meeting %s: %s", meeting_id, exc)
