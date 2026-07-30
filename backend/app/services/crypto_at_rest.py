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

# User-facing copy reused by API/FE (keep stable for UI matching).
DECRYPTION_FAILED_MESSAGE = "Decryption failed / Data corrupted"


class CryptoAtRestError(RuntimeError):
    """Base error for encryption-at-rest failures."""


class DecryptionError(CryptoAtRestError):
    """Ciphertext could not be decrypted (wrong key, missing key, or corrupt blob)."""


class EncryptionError(CryptoAtRestError):
    """Plaintext could not be encrypted with the configured key."""


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


def is_encrypted_blob(data: bytes | None) -> bool:
    """True when ``data`` carries the Smart Meeting ciphertext header."""
    return bool(data) and data.startswith(_HEADER)


def encrypt_bytes(plain: bytes) -> bytes:
    """Encrypt ``plain`` when a key is configured; otherwise return as-is."""
    if not plain:
        return plain
    f = _fernet()
    if f is None:
        return plain
    try:
        return _HEADER + f.encrypt(plain)
    except Exception as exc:
        raise EncryptionError(
            f"Could not encrypt audio blob: {exc}"
        ) from exc


def decrypt_bytes(data: bytes) -> bytes:
    """Decrypt SMENC1 payloads; pass through plaintext (legacy) blobs.

    Raises:
        DecryptionError: Encrypted payload cannot be decrypted (missing/wrong
            key or corrupted ciphertext). Never returns empty bytes to hide
            integrity failures — callers must handle the exception.
    """
    if not data:
        return data
    if not data.startswith(_HEADER):
        return data
    f = _fernet()
    if f is None:
        raise DecryptionError(
            f"{DECRYPTION_FAILED_MESSAGE}. "
            "Encrypted audio is present but DATA_ENCRYPTION_KEY is not configured "
            "(or the key is invalid)."
        )
    try:
        return f.decrypt(data[len(_HEADER) :])
    except Exception as exc:
        raise DecryptionError(
            f"{DECRYPTION_FAILED_MESSAGE}. "
            "The encryption key may be wrong after a rotation, or the stored "
            f"blob is damaged ({type(exc).__name__})."
        ) from exc


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
