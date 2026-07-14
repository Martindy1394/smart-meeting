"""Audio helpers: PCM (int16 LE) <-> float32, WAV persistence, and gain normalize.

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


def rms_level(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))


def peak_level(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.max(np.abs(samples)))


def normalize_audio(
    samples: np.ndarray,
    target_peak: float = 0.85,
    max_gain: float = 40.0,
) -> np.ndarray:
    """Peak-normalize quiet mic captures so Whisper/VAD can hear speech.

    Browser captures (or remote-forwarded mics) are sometimes extremely quiet,
    which makes VAD drop every frame and live captions stay blank.
    """
    if samples.size == 0:
        return samples
    peak = peak_level(samples)
    if peak < 1e-6:
        return samples
    gain = min(max_gain, target_peak / peak)
    if gain <= 1.05:
        return samples
    return np.clip(samples * gain, -1.0, 1.0).astype(np.float32)


def audio_dir() -> str:
    os.makedirs(settings.audio_storage_dir, exist_ok=True)
    return settings.audio_storage_dir


def save_wav(meeting_id: str, pcm_bytes: bytes) -> str:
    """Persist raw PCM bytes as a 16 kHz mono WAV file. Returns the path."""
    path = os.path.join(audio_dir(), f"{meeting_id}.wav")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(settings.audio_channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(settings.audio_sample_rate)
        wf.writeframes(pcm_bytes)
    return path


def wav_duration_seconds(pcm_bytes: bytes) -> float:
    samples = len(pcm_bytes) // 2
    return samples / float(settings.audio_sample_rate)
