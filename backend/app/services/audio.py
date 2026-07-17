"""Audio helpers: PCM (int16 LE) <-> float32, WAV persistence, and load for ASR.

The browser streams raw 16 kHz / 16-bit / mono PCM. Uploaded files are normalized
into that canonical format before Whisper ASR runs.

Non-WAV uploads are decoded via ``ffmpeg`` in a blocking subprocess — callers on
the asyncio event loop must wrap those paths with ``asyncio.to_thread``.
"""
from __future__ import annotations

import io
import logging
import os
import shutil
import subprocess
import wave

import numpy as np

from ..config import settings

logger = logging.getLogger("smart_meeting.audio")


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


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def decode_with_ffmpeg(data: bytes, *, timeout: float | None = None) -> np.ndarray:
    """Decode arbitrary audio bytes to mono float32 @ ``audio_sample_rate``.

    Uses a blocking ``ffmpeg`` subprocess. Never call this directly from an
    async request handler — wrap with ``asyncio.to_thread``.
    """
    if not data:
        raise AudioFormatError("Empty audio payload.")
    if not ffmpeg_available():
        raise AudioFormatError(
            "ffmpeg is not installed; only WAV/PCM uploads are supported."
        )

    target_rate = int(settings.audio_sample_rate)
    limit = float(timeout if timeout is not None else settings.ffmpeg_timeout_seconds)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(target_rate),
        "pipe:1",
    ]
    try:
        proc = subprocess.run(
            cmd,
            input=data,
            capture_output=True,
            timeout=max(5.0, limit),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioFormatError(
            f"Audio conversion timed out after {limit:g}s."
        ) from exc
    except OSError as exc:
        raise AudioFormatError(f"Failed to run ffmpeg: {exc}") from exc

    if proc.returncode != 0 or not proc.stdout:
        err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        detail = err or f"ffmpeg exited with code {proc.returncode}"
        raise AudioFormatError(f"Could not decode audio ({detail}).")

    samples = pcm16_to_float32(proc.stdout)
    logger.info(
        "audio.ffmpeg_decode bytes_in=%d samples_out=%d rate=%d",
        len(data),
        int(samples.size),
        target_rate,
    )
    return samples


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
    # Allow ffmpeg for unusual on-disk formats left by older uploads.
    if ffmpeg_available():
        return decode_with_ffmpeg(data)
    raise AudioFormatError(
        "Unsupported audio file. Upload a WAV recording (16-bit PCM recommended)."
    )


def save_uploaded_audio(meeting_id: str, data: bytes, filename: str = "") -> tuple[str, float]:
    """Normalize an uploaded audio file to canonical WAV and persist it.

    Returns ``(absolute_path, duration_seconds)``.

    Blocking (may invoke ffmpeg). Call via ``asyncio.to_thread`` from async routes.
    """
    name = (filename or "").lower()
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        samples, rate = decode_wav_bytes(data)
        mono_16k = normalize_for_asr(samples, rate)
    elif name.endswith(".pcm") or name.endswith(".raw"):
        mono_16k = pcm16_to_float32(data)
    else:
        # Prefer ffmpeg for mp3/m4a/ogg/webm and other containers.
        try:
            mono_16k = decode_with_ffmpeg(data)
        except AudioFormatError:
            # Last chance: some WAVs lack a standard extension.
            try:
                samples, rate = decode_wav_bytes(data)
                mono_16k = normalize_for_asr(samples, rate)
            except AudioFormatError as exc:
                raise AudioFormatError(
                    "Unsupported upload format. Please upload WAV, PCM, or a "
                    "common compressed audio file (mp3/m4a/ogg/webm)."
                ) from exc

    pcm_bytes = float32_to_pcm16(mono_16k)
    path = save_wav(meeting_id, pcm_bytes)
    return path, wav_duration_seconds(pcm_bytes)
