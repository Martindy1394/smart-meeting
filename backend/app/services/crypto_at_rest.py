"""Optional Fernet encryption-at-rest for audio blobs (disk + Redis).

When ``DATA_ENCRYPTION_KEY`` is set (url-safe Fernet key), finalized WAV and
Redis audio payloads are stored ciphertext-first. Live on-disk PCM remains
plaintext during capture for append performance, then is deleted after WAV
finalize — see docs/ENCRYPTION_AT_REST.md.

This is **not** end-to-end encryption; it protects data at rest on the server.
"""
from __future__ import annotations

import logging

from ..config import settings

logger = logging.getLogger("smart_meeting.crypto")

_HEADER = b"SMENC1\n"


def _fernet():
    key = (settings.data_encryption_key or "").strip()
    if not key:
        return None
    try:
        from cryptography.fernet import Fernet

        return Fernet(key.encode("utf-8") if isinstance(key, str) else key)
    except Exception as exc:
        logger.error("Invalid DATA_ENCRYPTION_KEY (%s); encryption disabled.", exc)
        return None


def encryption_enabled() -> bool:
    return _fernet() is not None


def encrypt_bytes(plain: bytes) -> bytes:
    """Encrypt ``plain`` when a key is configured; otherwise return as-is."""
    if not plain:
        return plain
    f = _fernet()
    if f is None:
        return plain
    return _HEADER + f.encrypt(plain)


def decrypt_bytes(data: bytes) -> bytes:
    """Decrypt SMENC1 payloads; pass through plaintext (legacy) blobs."""
    if not data:
        return data
    if not data.startswith(_HEADER):
        return data
    f = _fernet()
    if f is None:
        raise RuntimeError(
            "Encrypted audio is present but DATA_ENCRYPTION_KEY is not configured."
        )
    return f.decrypt(data[len(_HEADER) :])


def status() -> dict:
    return {
        "enabled": encryption_enabled(),
        "protects": ["finalized_wav_disk", "redis_wav", "redis_pcm_rolling"]
        if encryption_enabled()
        else [],
        "not_yet": [
            "live_disk_pcm_during_capture",
            "sqlite_row_encryption",
            "client_e2e",
        ],
    }
