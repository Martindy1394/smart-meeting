"""Voice activity detection gate before Whisper.

Primary backend: ``webrtcvad`` when installed. Fallback: energy + zero-crossing
rate gate that works with only numpy (always available). Silent / near-silent
chunks return ``has_speech=False`` so callers skip Whisper entirely — the main
source of hallucinated captions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..config import settings

logger = logging.getLogger("smart_meeting.vad")


@dataclass
class VadResult:
    has_speech: bool
    speech_ratio: float = 0.0
    backend: str = "energy"
    reason: str = ""


def _pcm_float_to_int16_bytes(pcm: np.ndarray) -> bytes:
    clipped = np.clip(pcm.astype(np.float32), -1.0, 1.0)
    ints = (clipped * 32767.0).astype("<i2")
    return ints.tobytes()


def _energy_vad(pcm: np.ndarray, *, sample_rate: int, min_rms: float) -> VadResult:
    if pcm is None or len(pcm) == 0:
        return VadResult(False, 0.0, "energy", "empty")
    x = pcm.astype(np.float32, copy=False)
    rms = float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
    if rms < min_rms:
        return VadResult(False, 0.0, "energy", f"rms={rms:.5f}<{min_rms}")
    # Zero-crossing rate: pure silence / DC bias has very low ZCR; speech is mid.
    signs = np.sign(x)
    signs[signs == 0] = 1
    zcr = float(np.mean(signs[:-1] != signs[1:])) if x.size > 1 else 0.0
    # Very low ZCR + modest RMS often = hum / AGC noise floor.
    if zcr < 0.01 and rms < min_rms * 3:
        return VadResult(False, 0.0, "energy", f"low_zcr={zcr:.4f}")
    return VadResult(True, 1.0, "energy", f"rms={rms:.5f}")


def _webrtc_vad(pcm: np.ndarray, *, sample_rate: int, aggressiveness: int) -> VadResult | None:
    try:
        import webrtcvad  # type: ignore
    except Exception:
        return None
    # webrtcvad supports 8/16/32/48 kHz; we use 16 kHz mono int16 frames of 20ms.
    if sample_rate not in (8000, 16000, 32000, 48000):
        return None
    try:
        vad = webrtcvad.Vad(int(max(0, min(3, aggressiveness))))
        raw = _pcm_float_to_int16_bytes(pcm)
        frame_ms = 20
        bytes_per_frame = int(sample_rate * frame_ms / 1000) * 2
        if bytes_per_frame <= 0 or len(raw) < bytes_per_frame:
            return VadResult(False, 0.0, "webrtcvad", "too_short")
        speech = 0
        total = 0
        for i in range(0, len(raw) - bytes_per_frame + 1, bytes_per_frame):
            frame = raw[i : i + bytes_per_frame]
            total += 1
            if vad.is_speech(frame, sample_rate):
                speech += 1
        if total == 0:
            return VadResult(False, 0.0, "webrtcvad", "no_frames")
        ratio = speech / float(total)
        min_ratio = float(getattr(settings, "vad_min_speech_ratio", 0.15) or 0.15)
        ok = ratio >= min_ratio
        return VadResult(
            ok,
            ratio,
            "webrtcvad",
            f"speech_frames={speech}/{total} ratio={ratio:.3f}",
        )
    except Exception as exc:
        logger.debug("webrtcvad failed (%s); using energy fallback", exc)
        return None


def detect_speech(
    pcm: np.ndarray,
    *,
    sample_rate: int | None = None,
    live: bool = False,
) -> VadResult:
    """Return whether ``pcm`` contains enough speech to send to Whisper."""
    if not bool(getattr(settings, "vad_enabled", True)):
        return VadResult(True, 1.0, "disabled", "vad_disabled")
    sr = int(sample_rate or settings.audio_sample_rate or 16000)
    min_rms = (
        float(getattr(settings, "vad_live_min_rms", 0.006) or 0.006)
        if live
        else float(getattr(settings, "vad_final_min_rms", 0.004) or 0.004)
    )
    backend = (getattr(settings, "vad_backend", "auto") or "auto").strip().lower()
    if backend in {"auto", "webrtcvad", "webrtc"}:
        result = _webrtc_vad(
            pcm,
            sample_rate=sr,
            aggressiveness=int(getattr(settings, "vad_aggressiveness", 2) or 2),
        )
        if result is not None:
            if not result.has_speech:
                logger.info("vad.skip backend=%s %s live=%s", result.backend, result.reason, live)
            return result
    return _energy_vad(pcm, sample_rate=sr, min_rms=min_rms)


def speech_mask_ratio(pcm: np.ndarray, *, sample_rate: int | None = None) -> float:
    """Fraction of frames classified as speech (0–1)."""
    result = detect_speech(pcm, sample_rate=sample_rate, live=False)
    return float(result.speech_ratio if result.backend == "webrtcvad" else (1.0 if result.has_speech else 0.0))
