"""Audio helpers: PCM (int16 LE) <-> float32, and WAV persistence.

The browser streams raw 16 kHz / 16-bit / mono PCM.  We keep everything in that
canonical format so no server-side decoding (WebM/Opus) is required.
"""
from __future__ import annotations

import os
import wave

import numpy as np

from ..config import settings


def pcm16_to_float32(data: bytes) -> np.ndarray:
    """Convert little-endian 16-bit PCM bytes to mono float32 in [-1, 1]."""
    if not data:
        return np.zeros(0, dtype=np.float32)
    # Guard against odd byte counts from partial chunks.
    if len(data) % 2:
        data = data[:-1]
    ints = np.frombuffer(data, dtype="<i2").astype(np.float32)
    return ints / 32768.0


def float32_to_pcm16(samples: np.ndarray) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def audio_dir() -> str:
    path = settings.audio_storage_dir
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)
    return path


def save_wav(meeting_id: str, pcm_bytes: bytes) -> str:
    """Persist raw PCM bytes as a 16 kHz mono WAV file. Returns an absolute path."""
    path = os.path.join(audio_dir(), f"{meeting_id}.wav")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(settings.audio_channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(settings.audio_sample_rate)
        wf.writeframes(pcm_bytes)
    return os.path.abspath(path)


def wav_duration_seconds(pcm_bytes: bytes) -> float:
    samples = len(pcm_bytes) // 2
    return samples / float(settings.audio_sample_rate)
