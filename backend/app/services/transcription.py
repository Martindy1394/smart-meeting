"""Whisper transcription service — the ``TranscribeAudio`` integration.

Implements the two-pass pipeline:

* **Live pass** — fast Whisper model on overlapping 10s windows (5s hop),
  locked to ``language="tl"`` and ``task="transcribe"``.
* **Final pass** — fine-tuned Hiligaynon / Philippine Whisper checkpoint from
  Hugging Face (``WHISPER_HILIGAYNON_MODEL``), also with ``task="transcribe"``
  and ``language="tl"``.

``faster_whisper`` / ``transformers`` are imported lazily so the rest of the
application runs even when model weights are not installed.
"""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass

import numpy as np

from ..config import settings

logger = logging.getLogger("smart_meeting.transcription")

# Soft VAD for the faster-whisper fallback final path.
_FINAL_VAD_PARAMS = {
    "onset": 0.35,
    "offset": 0.25,
    "min_speech_duration_ms": 150,
    "min_silence_duration_ms": 700,
    "speech_pad_ms": 400,
}

# Forced Whisper decode settings (never leave language as None).
_WHISPER_TASK = "transcribe"
_WHISPER_LANGUAGE = "tl"

# Keep this short and non-imperative — Whisper sometimes echoes prompts
# that look like transcript instructions (e.g. "Transcribe the spoken words…").
_INITIAL_PROMPT = (
    "Board meeting discussion in Hiligaynon, Filipino, or English."
)


class TranscriptionUnavailable(RuntimeError):
    """Raised when the Whisper backend cannot be loaded."""


@dataclass
class Segment:
    text: str
    start: float
    end: float


class _ModelCache:
    """Lazily loads and caches Whisper backends, thread-safely."""

    def __init__(self) -> None:
        self._fw_models: dict[str, object] = {}
        self._hf_pipelines: dict[str, object] = {}
        self._lock = threading.Lock()

    def get_faster_whisper(self, model_size: str):
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise TranscriptionUnavailable(
                "faster-whisper is not installed. Install backend ML deps: "
                "pip install -r requirements-ml.txt"
            ) from exc

        with self._lock:
            if model_size not in self._fw_models:
                logger.info(
                    "Loading faster-whisper model '%s' (device=%s)",
                    model_size,
                    settings.whisper_device,
                )
                self._fw_models[model_size] = WhisperModel(
                    model_size,
                    device=settings.whisper_device,
                    compute_type=settings.whisper_compute_type,
                )
            return self._fw_models[model_size]

    def get_hf_pipeline(self, model_id: str):
        """Load a Hugging Face fine-tuned Whisper ASR pipeline."""
        try:
            import torch  # type: ignore
            from transformers import pipeline  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise TranscriptionUnavailable(
                "transformers/torch are not installed. Install backend ML deps: "
                "pip install -r requirements-ml.txt"
            ) from exc

        with self._lock:
            if model_id not in self._hf_pipelines:
                device = 0 if settings.whisper_device == "cuda" and torch.cuda.is_available() else -1
                dtype = torch.float16 if device >= 0 else torch.float32
                logger.info(
                    "Loading fine-tuned Whisper ASR '%s' via transformers (device=%s)",
                    model_id,
                    "cuda" if device >= 0 else "cpu",
                )
                self._hf_pipelines[model_id] = pipeline(
                    "automatic-speech-recognition",
                    model=model_id,
                    device=device,
                    torch_dtype=dtype,
                    chunk_length_s=30,
                    stride_length_s=5,
                    return_timestamps=True,
                )
            return self._hf_pipelines[model_id]


_cache = _ModelCache()


def is_available() -> bool:
    try:
        import faster_whisper  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def hiligaynon_model_id() -> str:
    """Hugging Face / local id used for the final Stop-Recording pass."""
    return (settings.whisper_hiligaynon_model or settings.whisper_final_model).strip()


_REPEAT_WORD_RE = re.compile(r"\b(\w+)(?:\s+\1){3,}\b", re.IGNORECASE)
_STUTTER_RE = re.compile(r"([A-Za-zÀ-ÿ])(?:-\1){3,}", re.IGNORECASE)


def _collapse_hallucinations(text: str) -> str:
    """Collapse obvious Whisper repetition loops (e.g. 'mic mic mic…')."""
    prev = None
    while prev != text:
        prev = text
        text = _REPEAT_WORD_RE.sub(r"\1", text)
        text = _STUTTER_RE.sub(r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _forced_language(_requested: str | None) -> str:
    """Always return Tagalog (``tl``) — never leave Whisper language as None."""
    return _WHISPER_LANGUAGE


def transcribe_live(pcm: np.ndarray, language: str | None) -> list[Segment]:
    """Low-latency transcription of a live audio window.

    Uses overlapping windows upstream; decode is locked to Tagalog transcription.
    """
    if pcm is None or len(pcm) == 0:
        return []

    model = _cache.get_faster_whisper(settings.whisper_live_model)
    lang = _forced_language(language)
    segments, _info = model.transcribe(
        pcm,
        language=lang,
        task=_WHISPER_TASK,
        beam_size=3,
        best_of=3,
        temperature=0.0,
        vad_filter=False,
        condition_on_previous_text=False,
        without_timestamps=False,
        no_speech_threshold=0.2,
        compression_ratio_threshold=3.2,
        log_prob_threshold=-1.5,
        # No initial_prompt on live windows — short/noisy chunks often echo it.
        initial_prompt=None,
    )
    out: list[Segment] = []
    for s in segments:
        text = _collapse_hallucinations((s.text or "").strip())
        if text and any(ch.isalnum() for ch in text):
            out.append(Segment(text=text, start=s.start, end=s.end))
    return out


def _clean_caption(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _raw_tokens(text: str) -> list[str]:
    return _clean_caption(text).split() if _clean_caption(text) else []


def _norm_token(token: str) -> str:
    """Lowercase token with punctuation stripped for overlap matching."""
    return re.sub(r"[^\w]+", "", token, flags=re.UNICODE).lower()


def _norm_tokens(text: str) -> list[str]:
    out: list[str] = []
    for tok in _raw_tokens(text):
        n = _norm_token(tok)
        if n:
            out.append(n)
    return out


def _token_overlap_size(left: list[str], right: list[str], *, min_size: int = 3) -> int:
    """Longest suffix(left) == prefix(right) overlap, requiring ``min_size`` tokens."""
    max_overlap = min(len(left), len(right), 60)
    for size in range(max_overlap, min_size - 1, -1):
        if left[-size:] == right[:size]:
            return size
    return 0


def _novel_suffix_from_window(previous_window: str, current_window: str) -> str:
    """Return only the new words from an overlapping Whisper window.

    With a 10s window / 5s hop, Whisper re-transcribes the shared half every
    time. Using the previous window as the overlap anchor lets us append only
    the hop's new words instead of replacing / truncating the caption.
    """
    cur_tokens = _raw_tokens(current_window)
    if not cur_tokens:
        return ""
    prev_n = _norm_tokens(previous_window)
    if not prev_n:
        return _clean_caption(current_window)

    # Parallel list of (original_index, norm) for current window tokens.
    cur_pairs: list[tuple[int, str]] = []
    for idx, tok in enumerate(cur_tokens):
        n = _norm_token(tok)
        if n:
            cur_pairs.append((idx, n))
    if not cur_pairs:
        return ""

    cur_n = [n for _, n in cur_pairs]
    overlap = _token_overlap_size(prev_n, cur_n, min_size=3)

    if overlap == 0:
        # Fuzzy: align with SequenceMatcher and keep a block near the end of
        # the previous window (typical of sliding-window re-transcription).
        from difflib import SequenceMatcher

        matcher = SequenceMatcher(a=prev_n, b=cur_n, autojunk=False)
        best_b_end = 0
        best_size = 0
        for block in matcher.get_matching_blocks():
            if block.size < 3:
                continue
            # Prefer matches that touch the end of the previous window.
            if block.a + block.size >= len(prev_n) - 2 and block.size >= best_size:
                best_size = block.size
                best_b_end = block.b + block.size
        if best_size >= 3:
            if best_b_end >= len(cur_pairs):
                return ""
            start = cur_pairs[best_b_end][0]
            return " ".join(cur_tokens[start:]).strip()
        # No reliable overlap — keep roughly the newest half (the hop region)
        # so we still grow instead of re-pasting the whole window.
        cut = max(1, int(round(len(cur_tokens) * 0.5)))
        return " ".join(cur_tokens[-cut:]).strip()

    if overlap >= len(cur_pairs):
        return ""
    start = cur_pairs[overlap][0]
    return " ".join(cur_tokens[start:]).strip()


# Only the caption/window *tail* is needed for overlap — critical for 8h+ meetings
# where the full live caption can be tens of thousands of tokens.
_CAPTION_TAIL_TOKENS = 80


def _tail_text(text: str, max_tokens: int = _CAPTION_TAIL_TOKENS) -> str:
    toks = _raw_tokens(text)
    if len(toks) <= max_tokens:
        return _clean_caption(text)
    return " ".join(toks[-max_tokens:])


def _append_novel(caption: str, novel: str) -> str:
    """Append novel words to the caption without eating earlier text."""
    prev = _clean_caption(caption)
    addition = _clean_caption(novel)
    if not addition:
        return prev
    if not prev:
        return addition

    # Compare only against the caption tail (keeps merge O(1) for long meetings).
    prev_n = _norm_tokens(_tail_text(prev))
    add_tokens = _raw_tokens(addition)
    add_pairs: list[tuple[int, str]] = []
    for idx, tok in enumerate(add_tokens):
        n = _norm_token(tok)
        if n:
            add_pairs.append((idx, n))
    add_n = [n for _, n in add_pairs]
    if add_n and _token_overlap_size(prev_n, add_n, min_size=1) >= len(add_n):
        return prev

    overlap = _token_overlap_size(prev_n, add_n, min_size=1)
    if overlap and overlap < len(add_pairs):
        start = add_pairs[overlap][0]
        rest = " ".join(add_tokens[start:]).strip()
        if not rest:
            return prev
        return f"{prev} {rest}".strip()
    if overlap >= len(add_pairs):
        return prev
    return f"{prev} {addition}".strip()


def merge_live_caption(
    previous: str,
    window_text: str,
    *,
    previous_window: str | None = None,
) -> str:
    """Merge an overlapping-window transcript into the running live caption.

    Invariant: the returned caption never shrinks versus ``previous``. Overlap
    audio is deduplicated by comparing against the previous window (preferred)
    or the caption tail, then only new words are appended.
    """
    prev = _clean_caption(previous)
    cur = _clean_caption(window_text)
    if not cur:
        return prev
    if not prev:
        return cur

    # Prefer window-to-window novel extraction (correct for 10s/5s overlap).
    if previous_window:
        novel = _novel_suffix_from_window(previous_window, cur)
    else:
        novel = ""

    # Fallback: only the caption *tail* — never scan multi-hour history.
    if not novel:
        novel = _novel_suffix_from_window(_tail_text(prev), cur)

    # Last resort: if novel detection failed entirely, append the window only
    # when it is not already at the caption end (avoid wipe/replace).
    if not novel:
        cur_n = _norm_tokens(cur)
        prev_n = _norm_tokens(_tail_text(prev))
        if cur_n and any(
            prev_n[i : i + len(cur_n)] == cur_n
            for i in range(0, max(1, len(prev_n) - len(cur_n) + 1))
        ):
            return prev
        novel = cur

    merged = _append_novel(prev, novel)

    # Hard monotonic guard — never let dedupe erase earlier speech.
    if len(_raw_tokens(merged)) < len(_raw_tokens(prev)):
        logger.warning(
            "Live caption merge tried to shrink (%d -> %d tokens); keeping previous.",
            len(_raw_tokens(prev)),
            len(_raw_tokens(merged)),
        )
        return prev
    return merged


def _segments_to_list(segments) -> list[Segment]:
    out: list[Segment] = []
    for s in segments:
        text = _collapse_hallucinations((s.text or "").strip())
        if text:
            out.append(Segment(text=text, start=s.start, end=s.end))
    return out


def _audio_to_float32(audio_source) -> np.ndarray:
    if isinstance(audio_source, np.ndarray):
        return audio_source.astype(np.float32, copy=False)
    from . import audio as audio_svc

    return audio_svc.load_audio_float32(str(audio_source))


def _transcribe_final_hf(audio_source, language: str | None) -> list[Segment]:
    """Final pass with a Hugging Face fine-tuned Whisper checkpoint."""
    model_id = hiligaynon_model_id()
    pipe = _cache.get_hf_pipeline(model_id)
    samples = _audio_to_float32(audio_source)
    if samples.size == 0:
        return []

    lang = _forced_language(language)
    logger.info(
        "Final Whisper ASR with fine-tuned model '%s' (language=%s, task=%s)",
        model_id,
        lang,
        _WHISPER_TASK,
    )
    result = pipe(
        {"array": samples, "sampling_rate": int(settings.audio_sample_rate)},
        generate_kwargs={
            "language": lang,
            "task": _WHISPER_TASK,
        },
        return_timestamps=True,
    )

    chunks = result.get("chunks") if isinstance(result, dict) else None
    if chunks:
        out: list[Segment] = []
        for i, chunk in enumerate(chunks):
            text = _collapse_hallucinations((chunk.get("text") or "").strip())
            if not text:
                continue
            ts = chunk.get("timestamp") or (None, None)
            start = float(ts[0] or 0.0)
            end = float(ts[1] if ts[1] is not None else start)
            out.append(Segment(text=text, start=start, end=end))
        if out:
            return out

    text = ""
    if isinstance(result, dict):
        text = _collapse_hallucinations((result.get("text") or "").strip())
    elif isinstance(result, str):
        text = _collapse_hallucinations(result.strip())
    if not text:
        return []
    duration = float(samples.size) / float(settings.audio_sample_rate)
    return [Segment(text=text, start=0.0, end=duration)]


def _transcribe_final_faster_whisper(audio_source, language: str | None) -> list[Segment]:
    """Fallback final pass via faster-whisper (stock or converted model id)."""
    model = _cache.get_faster_whisper(settings.whisper_final_model)
    lang = _forced_language(language)
    segments, info = model.transcribe(
        audio_source if isinstance(audio_source, str) else _audio_to_float32(audio_source),
        language=lang,
        task=_WHISPER_TASK,
        beam_size=5,
        best_of=5,
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        vad_filter=True,
        vad_parameters=_FINAL_VAD_PARAMS,
        condition_on_previous_text=False,
        without_timestamps=False,
        initial_prompt=_INITIAL_PROMPT,
        no_speech_threshold=0.55,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
    )
    detected = getattr(info, "language", None)
    if detected:
        logger.info("faster-whisper final language reported: %s", detected)
    return _segments_to_list(segments)


def _looks_like_hf_repo(model_id: str) -> bool:
    mid = (model_id or "").strip()
    return "/" in mid or mid.startswith(".") or mid.startswith("/")


def _transcribe_final_once(audio_source, language: str | None) -> list[Segment]:
    """Single-shot final ASR (HF fine-tune preferred, faster-whisper fallback)."""
    model_id = hiligaynon_model_id()
    if _looks_like_hf_repo(model_id):
        try:
            return _transcribe_final_hf(audio_source, language)
        except TranscriptionUnavailable:
            raise
        except Exception as exc:
            logger.exception(
                "Fine-tuned Hiligaynon Whisper '%s' failed (%s); "
                "falling back to faster-whisper '%s'.",
                model_id,
                exc,
                settings.whisper_final_model,
            )
    return _transcribe_final_faster_whisper(audio_source, language)


def _merge_chunk_segments(
    all_segments: list[Segment],
    chunk_segments: list[Segment],
    *,
    time_offset: float,
) -> list[Segment]:
    """Append chunk segments with timeline offset, dropping overlap duplicates."""
    shifted = [
        Segment(text=s.text, start=s.start + time_offset, end=s.end + time_offset)
        for s in chunk_segments
        if (s.text or "").strip()
    ]
    if not all_segments:
        return shifted
    if not shifted:
        return all_segments

    # Drop leading shifted segments that largely repeat the previous chunk tail.
    tail_text = " ".join(s.text for s in all_segments[-6:]).strip()
    tail_n = _norm_tokens(_tail_text(tail_text, 40))
    keep_from = 0
    for i, seg in enumerate(shifted):
        seg_n = _norm_tokens(seg.text)
        if not seg_n:
            keep_from = i + 1
            continue
        overlap = _token_overlap_size(tail_n, seg_n, min_size=min(3, len(seg_n)))
        if overlap >= max(3, int(len(seg_n) * 0.7)):
            keep_from = i + 1
            tail_n = _norm_tokens(
                _tail_text(" ".join(s.text for s in all_segments[-6:] + shifted[: keep_from]), 40)
            )
            continue
        break
    return all_segments + shifted[keep_from:]


def transcribe_final(audio_source, language: str | None) -> list[Segment]:
    """Full-accuracy transcription for Stop Recording / re-transcribe / upload.

    Prefer the configured fine-tuned Hiligaynon (or PH-dialect) Hugging Face
    Whisper model. Fall back to faster-whisper if that checkpoint cannot load.
    Always forces ``language="tl"`` and ``task="transcribe"``.

    Long board-meeting audio (hours) is processed in overlapping chunks so the
    final pass stays within model/memory limits.
    """
    samples = _audio_to_float32(audio_source)
    if samples.size == 0:
        return []

    sr = int(settings.audio_sample_rate)
    duration = float(samples.size) / float(sr)
    chunk_s = max(60.0, float(settings.whisper_final_chunk_seconds))
    overlap_s = max(0.0, min(float(settings.whisper_final_chunk_overlap_seconds), chunk_s / 2))

    # Short recordings: single pass.
    if duration <= chunk_s * 1.25:
        return _transcribe_final_once(samples, language)

    hop_s = max(1.0, chunk_s - overlap_s)
    chunk_samples = int(chunk_s * sr)
    hop_samples = int(hop_s * sr)
    logger.info(
        "Chunked final ASR for %.1f min audio (chunk=%.0fs overlap=%.0fs)",
        duration / 60.0,
        chunk_s,
        overlap_s,
    )

    all_segments: list[Segment] = []
    offset = 0
    chunk_idx = 0
    while offset < samples.size:
        end = min(samples.size, offset + chunk_samples)
        piece = samples[offset:end]
        if piece.size < int(0.4 * sr):
            break
        chunk_idx += 1
        time_offset = offset / float(sr)
        logger.info(
            "Final ASR chunk %d (%.1f–%.1f min)",
            chunk_idx,
            time_offset / 60.0,
            end / float(sr) / 60.0,
        )
        try:
            segs = _transcribe_final_once(piece, language)
        except Exception:
            logger.exception("Final ASR chunk %d failed; continuing", chunk_idx)
            segs = []
        all_segments = _merge_chunk_segments(
            all_segments, segs, time_offset=time_offset
        )
        if end >= samples.size:
            break
        offset += hop_samples

    return all_segments
