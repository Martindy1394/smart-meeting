"""Optional NVIDIA NeMo RNN-T (FastConformer-Hybrid) live ASR backend.

Why this exists
---------------
Whisper is strong for *final* offline passes but is not a true streaming
transducer. Overlapping 10s live windows re-decode a lot of audio. An RNN-T /
FastConformer-Hybrid model emits tokens as audio arrives and is a better fit
for low-latency live captions.

Default product path still uses faster-whisper when NeMo is not installed.
When ``WHISPER_LIVE_BACKEND=auto|rnnt`` and NeMo + the Tagalog FastConformer
checkpoint are available, Philippine / Hiligaynon-biased meetings use RNNT for
*live* captions only. Final re-transcribe stays on Whisper / PH-medium.

Recommended checkpoint
----------------------
``NCSpeech/stt_tl_fastconformer_hybrid_large`` — Tagalog/Filipino FastConformer
hybrid RNNT+CTC (~115M). Closest production RNNT for PH speech today; no public
Hiligaynon RNNT exists yet (fine-tune this base on Ilonggo board audio later).
"""
from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
import wave
from typing import Any

import numpy as np

from ..config import settings
from .transcription import LanguageDetection, Segment, effective_asr_language, is_philippine_language

logger = logging.getLogger("smart_meeting.rnnt")

_lock = threading.RLock()
_model: Any | None = None
_model_id: str | None = None
_load_error: str | None = None
_unavailable_logged = False


def live_backend() -> str:
    return (settings.whisper_live_backend or "auto").strip().lower()


def rnnt_model_id() -> str:
    return (
        (settings.rnnt_live_model or "").strip()
        or "NCSpeech/stt_tl_fastconformer_hybrid_large"
    )


def should_use_rnnt_live(language: str | None) -> bool:
    """True when live captions should prefer RNN-T for this meeting language."""
    mode = live_backend()
    if mode in {"whisper", "fw", "faster-whisper", "off", "none"}:
        return False
    if mode not in {"auto", "rnnt", "nemo", "transducer"}:
        return False
    # RNNT checkpoint is Tagalog/PH-oriented — use for Hiligaynon-biased auto too.
    return is_philippine_language(effective_asr_language(language))


def is_available() -> bool:
    """True when NeMo can be imported (model may still need a first download)."""
    global _unavailable_logged, _load_error
    if _load_error:
        return False
    try:
        import nemo.collections.asr  # noqa: F401
        return True
    except Exception as exc:
        _load_error = str(exc)
        if not _unavailable_logged:
            logger.info(
                "NeMo RNN-T live ASR unavailable (%s). "
                "Install optional deps: pip install -r requirements-rnnt.txt",
                exc,
            )
            _unavailable_logged = True
        return False


def status() -> dict:
    """Operator-facing RNNT status for /api/health."""
    importable = False
    try:
        import nemo.collections.asr  # noqa: F401

        importable = True
    except Exception:
        importable = False
    return {
        "configured_backend": live_backend(),
        "model": rnnt_model_id(),
        "nemo_importable": importable,
        "available_for_live": is_available(),
        "model_loaded": _model is not None,
        "load_error": _load_error,
    }


def _pcm_to_temp_wav(pcm: np.ndarray, sample_rate: int) -> str:
    path = tempfile.mktemp(suffix=".wav", prefix="rnnt_live_")
    clipped = np.clip(pcm.astype(np.float32), -1.0, 1.0)
    ints = (clipped * 32767.0).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(ints.tobytes())
    return path


def _resolve_nemo_path(mid: str) -> str:
    """Resolve a local .nemo path or download from Hugging Face."""
    from huggingface_hub import hf_hub_download

    if mid.endswith(".nemo"):
        if not os.path.isfile(mid):
            raise FileNotFoundError(f"NeMo checkpoint not found: {mid}")
        return mid
    if os.path.isdir(mid):
        for name in os.listdir(mid):
            if name.endswith(".nemo"):
                return os.path.join(mid, name)
        raise FileNotFoundError(f"No .nemo file in {mid}")

    candidates = [
        "stt_tl_fastconformer_hybrid_large.nemo",
        f"{mid.split('/')[-1]}.nemo",
    ]
    last_exc: Exception | None = None
    for filename in candidates:
        try:
            return hf_hub_download(repo_id=mid, filename=filename)
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"Could not download .nemo for {mid}: {last_exc}")


def _load_model():
    global _model, _model_id, _load_error
    mid = rnnt_model_id()
    with _lock:
        if _model is not None and _model_id == mid:
            return _model
        if _load_error and _model_id == mid:
            raise RuntimeError(_load_error)
        try:
            import nemo.collections.asr as nemo_asr
        except Exception as exc:
            _load_error = str(exc)
            _model_id = mid
            raise RuntimeError(_load_error) from exc

        t0 = time.perf_counter()
        logger.info("Loading NeMo RNN-T live model '%s'…", mid)
        try:
            nemo_path = _resolve_nemo_path(mid)
            model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
                nemo_path
            )
            model.eval()
            # Prefer transducer decoding when hybrid.
            try:
                model.change_decoding_strategy(decoder_type="rnnt")
            except Exception:
                pass
            _model = model
            _model_id = mid
            _load_error = None
            logger.info(
                "NeMo RNN-T ready (%s) in %.1fs",
                mid,
                time.perf_counter() - t0,
            )
            return _model
        except Exception as exc:
            _load_error = str(exc)
            _model = None
            _model_id = mid
            logger.exception("Failed to load NeMo RNN-T model '%s'", mid)
            raise


def transcribe_live(
    pcm: np.ndarray, language: str | None = None
) -> tuple[list[Segment], LanguageDetection | None]:
    """Run RNN-T on one live PCM window. Raises on hard failure (caller falls back)."""
    if pcm is None or len(pcm) == 0:
        return [], None
    model = _load_model()
    sr = int(settings.audio_sample_rate or 16000)
    duration = float(len(pcm)) / float(sr)
    wav_path = _pcm_to_temp_wav(pcm, sr)
    t0 = time.perf_counter()
    try:
        with _lock:
            # NeMo 1.x/2.x: list of paths → list of strings / hypotheses.
            raw = model.transcribe([wav_path], verbose=False)
        text = ""
        if raw is None:
            text = ""
        elif isinstance(raw, list) and raw:
            first = raw[0]
            text = getattr(first, "text", None) or (first if isinstance(first, str) else str(first))
        elif isinstance(raw, str):
            text = raw
        text = (text or "").strip()
        if not text:
            return [], LanguageDetection(
                language="tl",
                confidence=None,
                detected_by="rnnt",
            )
        segs = [Segment(text=text, start=0.0, end=duration)]
        logger.info(
            "asr.live.rnnt duration_ms=%d chars=%d model=%s lang=%s",
            int((time.perf_counter() - t0) * 1000),
            len(text),
            rnnt_model_id(),
            language,
        )
        return segs, LanguageDetection(
            language="tl",
            confidence=None,
            detected_by="rnnt",
        )
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass
