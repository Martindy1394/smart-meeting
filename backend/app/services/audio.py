"""Audio helpers: PCM (int16 LE) <-> float32, WAV persistence, and load for ASR.

The browser streams raw 16 kHz / 16-bit / mono PCM. Uploaded files are normalized
into that canonical format before Whisper ASR runs.
"""
from __future__ import annotations

import io
import os
import wave

import numpy as np

from ..config import settings


class AudioFormatError(ValueError):
    """Raised when uploaded audio cannot be decoded for Whisper ASR."""


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


def build_wav_bytes(pcm_bytes: bytes) -> bytes:
    """Encode raw PCM as a 16 kHz mono WAV container in memory."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(settings.audio_channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(settings.audio_sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def save_wav(meeting_id: str, pcm_bytes: bytes) -> str:
    """Persist raw PCM bytes as a 16 kHz mono WAV file. Returns an absolute path.

    Also caches the WAV bytes in Redis memory storage when Redis is available.
    """
    wav_bytes = build_wav_bytes(pcm_bytes)
    path = os.path.join(audio_dir(), f"{meeting_id}.wav")
    with open(path, "wb") as fh:
        fh.write(wav_bytes)

    # Mirror into Redis so recorded audio lives in memory storage too.
    try:
        from . import redis_store

        redis_store.save_wav_bytes(meeting_id, wav_bytes)
    except Exception:
        pass

    return os.path.abspath(path)


def wav_duration_seconds(pcm_bytes: bytes) -> float:
    samples = len(pcm_bytes) // 2
    return samples / float(settings.audio_sample_rate)


def _resample_mono(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or samples.size == 0:
        return samples.astype(np.float32, copy=False)
    duration = samples.size / float(src_rate)
    target_len = max(1, int(round(duration * dst_rate)))
    x_old = np.linspace(0.0, 1.0, num=samples.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
    return np.interp(x_new, x_old, samples).astype(np.float32)


def decode_wav_bytes(data: bytes) -> tuple[np.ndarray, int]:
    """Decode a WAV container into mono float32 samples and sample rate."""
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
    except wave.Error as exc:
        raise AudioFormatError(f"Invalid WAV audio: {exc}") from exc

    if sampwidth == 1:
        ints = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        samples = (ints - 128.0) / 128.0
    elif sampwidth == 2:
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 4:
        samples = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise AudioFormatError(f"Unsupported WAV sample width: {sampwidth} bytes")

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    return samples.astype(np.float32), int(rate)


def normalize_for_asr(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Resample / shape audio to the Whisper ASR canonical rate (16 kHz mono)."""
    target = int(settings.audio_sample_rate)
    return _resample_mono(samples.astype(np.float32, copy=False), sample_rate, target)


def load_audio_float32(path: str) -> np.ndarray:
    """Load a stored WAV into float32 mono @ ``audio_sample_rate`` for Whisper."""
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        samples, rate = decode_wav_bytes(data)
        return normalize_for_asr(samples, rate)
    raise AudioFormatError(
        "Unsupported audio file. Upload a WAV recording (16-bit PCM recommended)."
    )


def save_uploaded_audio(meeting_id: str, data: bytes, filename: str = "") -> tuple[str, float]:
    """Normalize an uploaded audio file to canonical WAV and persist it.

    Returns ``(absolute_path, duration_seconds)``.
    """
    name = (filename or "").lower()
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        samples, rate = decode_wav_bytes(data)
    elif name.endswith(".pcm") or name.endswith(".raw"):
        samples = pcm16_to_float32(data)
        rate = settings.audio_sample_rate
    else:
        try:
            samples, rate = decode_wav_bytes(data)
        except AudioFormatError as exc:
            raise AudioFormatError(
                "Unsupported upload format. Please upload a WAV (or raw 16-bit PCM) file."
            ) from exc

    mono_16k = normalize_for_asr(samples, rate)
    pcm_bytes = float32_to_pcm16(mono_16k)
    path = save_wav(meeting_id, pcm_bytes)
    return path, wav_duration_seconds(pcm_bytes)
