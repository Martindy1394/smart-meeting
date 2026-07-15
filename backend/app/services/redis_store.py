"""Redis-backed memory storage for live meeting audio and session state.

All recorded PCM is appended to Redis during capture. Finalized WAV bytes are
also cached in Redis so audio lives in memory storage in addition to disk.
"""
from __future__ import annotations

import logging
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
            socket_timeout=5.0,
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


def _touch_ttl(client, meeting_id: str) -> None:
    ttl = int(settings.redis_audio_ttl_seconds)
    if ttl <= 0:
        return
    client.expire(_pcm_key(meeting_id), ttl)
    client.expire(_meta_key(meeting_id), ttl)
    client.expire(_wav_key(meeting_id), ttl)


def append_pcm(meeting_id: str, chunk: bytes) -> int:
    """Append a PCM chunk to the meeting's Redis audio buffer.

    Returns the total buffered byte length after append, or ``-1`` on failure.
    """
    if not chunk:
        return get_pcm_length(meeting_id)
    client = get_client()
    if client is None:
        return -1
    try:
        total = int(client.append(_pcm_key(meeting_id), chunk))
        _touch_ttl(client, meeting_id)
        return total
    except Exception as exc:
        logger.exception("Redis APPEND failed for meeting %s: %s", meeting_id, exc)
        return -1


def get_pcm(meeting_id: str) -> bytes:
    """Return the full recorded PCM for a meeting from Redis (or empty)."""
    client = get_client()
    if client is None:
        return b""
    try:
        data = client.get(_pcm_key(meeting_id))
        return data or b""
    except Exception as exc:
        logger.exception("Redis GET pcm failed for meeting %s: %s", meeting_id, exc)
        return b""


def get_pcm_length(meeting_id: str) -> int:
    client = get_client()
    if client is None:
        return 0
    try:
        return int(client.strlen(_pcm_key(meeting_id)))
    except Exception:
        return 0


def get_pcm_slice(meeting_id: str, start: int, end: int) -> bytes:
    """Inclusive Redis GETRANGE slice ``[start, end]`` (end inclusive)."""
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
    if not mapping:
        return
    try:
        client.hset(_meta_key(meeting_id), mapping=mapping)
        _touch_ttl(client, meeting_id)
    except Exception as exc:
        logger.exception("Redis HSET meta failed for meeting %s: %s", meeting_id, exc)


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

    return {
        "live_caption": _s(b"live_caption"),
        "previous_window": _s(b"previous_window"),
        "live_offset": _i(b"live_offset"),
        "seq": _i(b"seq"),
    }


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
