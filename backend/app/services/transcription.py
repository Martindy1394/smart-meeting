"""Whisper transcription service — the ``TranscribeAudio`` integration.

Implements the two-pass pipeline:

* **Live pass** — fast Whisper (or an optional CTranslate2 Hiligaynon fine-tune)
  on overlapping 10s windows (5s hop). App language ``hil`` maps to Whisper
  decode code ``tl`` (Whisper has no native Hiligaynon token).
* **Final pass** — prefers a fine-tuned Hiligaynon/PH transformers checkpoint
  (``WHISPER_HILIGAYNON_MODEL``) when ``WHISPER_FINAL_BACKEND=auto|huggingface``,
  with faster-whisper fallback for coverage / availability.

Fine-tuning itself is done *outside* this repo; point the env vars at your
checkpoint (see ``docs/FINE_TUNE_HILIGAYNON.md``).

``faster_whisper`` / ``transformers`` are imported lazily so the rest of the
application runs even when model weights are not installed.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from ..config import settings

logger = logging.getLogger("smart_meeting.transcription")

# Soft VAD only used when explicitly enabled for the final path.
_FINAL_VAD_PARAMS = {
    "onset": 0.25,
    "offset": 0.20,
    "min_speech_duration_ms": 100,
    "min_silence_duration_ms": 500,
    "speech_pad_ms": 500,
}

# Forced Whisper decode settings (never leave language as None).
_WHISPER_TASK = "transcribe"

# App / UI language labels that should use the Philippine ASR path.
_PH_LANGUAGE_LABELS = frozenset(
    {
        "hil",
        "hiligaynon",
        "ilonggo",
        "fil",
        "tl",
        "tagalog",
        "filipino",
    }
)

# Common Whisper spam / silence hallucinations to drop entirely.
_HALLUCINATION_PHRASES = (
    "subscribe to my channel",
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
    "like and subscribe",
    "see you next time",
    "transcribe the spoken words",
    "i'm sorry, but it's okay",
    "i'm sorry, but i can't do it",
    "i don't want to see you again",
    "i'm not sure if it's because of you",
)


class TranscriptionUnavailable(RuntimeError):
    """Raised when the Whisper backend cannot be loaded."""


@dataclass
class Segment:
    text: str
    start: float
    end: float


class _ModelCache:
    """Lazily loads Whisper backends with per-model infer locks + LRU eviction.

    faster-whisper and HF pipelines are not safe for concurrent calls on the
    *same* loaded model, but live (``small``) and final (``medium``) can run in
    parallel when they are distinct cache entries. Cache size is capped so
    long-running processes do not retain every model ever touched.
    """

    def __init__(self, max_models: int | None = None) -> None:
        self._fw_models: dict[str, object] = {}
        self._hf_pipelines: dict[str, object] = {}
        self._fw_locks: dict[str, threading.Lock] = {}
        self._hf_locks: dict[str, threading.Lock] = {}
        # Global LRU across both backends: ("fw"|"hf", model_id) -> None
        self._lru: OrderedDict[tuple[str, str], None] = OrderedDict()
        self._lock = threading.Lock()
        self._max_models = max(1, int(max_models or settings.whisper_model_cache_size))

    def _touch(self, kind: str, model_id: str) -> None:
        token = (kind, model_id)
        if token in self._lru:
            self._lru.move_to_end(token)
        else:
            self._lru[token] = None
        while len(self._lru) > self._max_models:
            old_kind, old_id = next(iter(self._lru))
            self._lru.popitem(last=False)
            if old_kind == "fw":
                self._fw_models.pop(old_id, None)
            else:
                self._hf_pipelines.pop(old_id, None)
            logger.info(
                "Evicted %s model cache entry '%s' (max_models=%d)",
                "faster-whisper" if old_kind == "fw" else "huggingface",
                old_id,
                self._max_models,
            )

    def get_faster_whisper(self, model_size: str):
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise TranscriptionUnavailable(
                "faster-whisper is not installed. Install backend ML deps: "
                "pip install -r requirements-ml.txt"
            ) from exc

        with self._lock:
            if model_size in self._fw_models:
                self._touch("fw", model_size)
                return self._fw_models[model_size]

        # Load outside the cache lock so concurrent warmups for different
        # models do not serialize on download / mmap.
        logger.info(
            "Loading faster-whisper model '%s' (device=%s)",
            model_size,
            settings.whisper_device,
        )
        model = WhisperModel(
            model_size,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
        with self._lock:
            existing = self._fw_models.get(model_size)
            if existing is not None:
                self._touch("fw", model_size)
                return existing
            self._fw_models[model_size] = model
            self._fw_locks.setdefault(model_size, threading.Lock())
            self._touch("fw", model_size)
            return model

    def fw_infer_lock(self, model_size: str) -> threading.Lock:
        """Per-model lock — live and final models can infer concurrently."""
        with self._lock:
            return self._fw_locks.setdefault(model_size, threading.Lock())

    def infer_lock(self) -> threading.Lock:
        """Backward-compatible alias: lock for the configured live model."""
        return self.fw_infer_lock(settings.whisper_live_model)

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
            if model_id in self._hf_pipelines:
                self._touch("hf", model_id)
                return self._hf_pipelines[model_id]

        device = 0 if settings.whisper_device == "cuda" and torch.cuda.is_available() else -1
        dtype = torch.float16 if device >= 0 else torch.float32
        logger.info(
            "Loading fine-tuned Whisper ASR '%s' via transformers (device=%s)",
            model_id,
            "cuda" if device >= 0 else "cpu",
        )
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=device,
            torch_dtype=dtype,
            chunk_length_s=30,
            stride_length_s=5,
            return_timestamps=True,
        )
        with self._lock:
            existing = self._hf_pipelines.get(model_id)
            if existing is not None:
                self._touch("hf", model_id)
                return existing
            self._hf_pipelines[model_id] = pipe
            self._hf_locks.setdefault(model_id, threading.Lock())
            self._touch("hf", model_id)
            return pipe

    def hf_infer_lock(self, model_id: str) -> threading.Lock:
        with self._lock:
            return self._hf_locks.setdefault(model_id, threading.Lock())

    def stats(self) -> dict:
        with self._lock:
            return {
                "max_models": self._max_models,
                "faster_whisper": list(self._fw_models.keys()),
                "huggingface": list(self._hf_pipelines.keys()),
                "lru": [f"{k}:{i}" for k, i in self._lru.keys()],
            }


_cache = _ModelCache()


def get_model_cache() -> _ModelCache:
    """Accessor for the process-wide model cache (testable / injectable)."""
    return _cache


def set_model_cache(cache: _ModelCache | None) -> _ModelCache:
    """Replace the process-wide cache (tests). Pass ``None`` to reset."""
    global _cache
    _cache = cache if cache is not None else _ModelCache()
    return _cache


def is_available() -> bool:
    try:
        import faster_whisper  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def warm_live_model() -> bool:
    """Load the live Whisper model into memory so the first caption is fast."""
    if not is_available():
        return False
    try:
        model_id = live_model_id(settings.whisper_default_language)
        get_model_cache().get_faster_whisper(model_id)
        logger.info("Live Whisper model '%s' warmed and ready.", model_id)
        return True
    except Exception as exc:
        logger.warning("Could not warm live Whisper model: %s", exc)
        return False


def hiligaynon_hf_candidates() -> list[str]:
    """Ordered HF/local Whisper ids for Hiligaynon final ASR.

    1. User fine-tune (``WHISPER_HILIGAYNON_FINE_TUNED_MODEL``) when set
    2. Philippine fallback (``WHISPER_HILIGAYNON_MODEL``, default
       ``rbcurzon/whisper-medium-ph``)
    """
    ordered = [
        (settings.whisper_hiligaynon_fine_tuned_model or "").strip(),
        (settings.whisper_hiligaynon_model or "").strip(),
    ]
    out: list[str] = []
    seen: set[str] = set()
    for mid in ordered:
        if not mid or mid in seen:
            continue
        seen.add(mid)
        out.append(mid)
    return out


def hiligaynon_model_id() -> str:
    """Primary Hugging Face / local id for the Hiligaynon final pass."""
    cands = hiligaynon_hf_candidates()
    if cands:
        return cands[0]
    return (settings.whisper_final_model or "medium").strip()


def is_philippine_language(language: str | None) -> bool:
    """True when the meeting language should use the Hiligaynon/PH ASR path."""
    lang = (language or settings.whisper_default_language or "").strip().lower()
    return lang in _PH_LANGUAGE_LABELS


def initial_prompt() -> str | None:
    prompt = (settings.whisper_initial_prompt or "").strip()
    return prompt or None


def live_model_id(language: str | None) -> str:
    """faster-whisper model for live captions (optional Hiligaynon CT2 fine-tune)."""
    custom = (settings.whisper_live_hiligaynon_model or "").strip()
    if custom and is_philippine_language(language):
        return custom
    return (settings.whisper_live_model or "small").strip()


def final_faster_model_id(language: str | None) -> str:
    """faster-whisper model for final fallback / FW-only backend."""
    # Prefer a CT2 export of the Hiligaynon fine-tune when configured.
    custom = (settings.whisper_live_hiligaynon_model or "").strip()
    if custom and is_philippine_language(language):
        return custom
    return (settings.whisper_final_model or "medium").strip()


def resolve_final_backend(language: str | None) -> str:
    """Normalize final ASR backend selection.

    ``auto`` prefers the fine-tuned Hugging Face Hiligaynon checkpoint for
    Philippine-language meetings, otherwise faster-whisper.
    """
    backend = (settings.whisper_final_backend or "auto").strip().lower()
    if backend in {"huggingface", "hf", "transformers"}:
        return "huggingface"
    if backend in {"faster-whisper", "fw", "ctranslate2", "ct2"}:
        return "faster-whisper"
    # auto / default
    if is_philippine_language(language) and hiligaynon_hf_candidates():
        return "huggingface"
    return "faster-whisper"


_REPEAT_WORD_RE = re.compile(r"\b(\w+)(?:\s+\1){3,}\b", re.IGNORECASE)
_STUTTER_RE = re.compile(r"([A-Za-zÀ-ÿ])(?:-\1){3,}", re.IGNORECASE)
_HYPHEN_LOOP_RE = re.compile(r"\b(\w+)(?:-\1){2,}\b", re.IGNORECASE)
# "papapapapa" / "nanananana" inside a single token.
_SYLLABLE_LOOP_RE = re.compile(r"\b([A-Za-zÀ-ÿ]{1,4}?)\1{3,}\b", re.IGNORECASE)


def _collapse_hallucinations(text: str) -> str:
    """Collapse obvious Whisper repetition loops (e.g. 'mic mic mic…')."""
    prev = None
    while prev != text:
        prev = text
        text = _REPEAT_WORD_RE.sub(r"\1", text)
        text = _STUTTER_RE.sub(r"\1", text)
        text = _HYPHEN_LOOP_RE.sub(r"\1", text)
        text = _SYLLABLE_LOOP_RE.sub(r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _forced_language(requested: str | None = None) -> str:
    """Map app language labels to a Whisper-supported decode code.

    Whisper has no native ``hil`` token. Hiligaynon / Filipino meetings use the
    closest supported Philippine code (default ``tl``).
    """
    app_lang = (requested or "").strip().lower()
    if app_lang in {"en", "english"}:
        return "en"
    # hil / fil / tl / tagalog → configured PH decode code
    lang = (settings.whisper_decode_language or "tl").strip().lower()
    return lang or "tl"


def _final_decode_language(requested: str | None) -> str | None:
    """Language for the final pass.

    Board meetings mix English + PH languages. Forcing ``tl`` caused Whisper to
    skip long spans of clearly spoken English. Default is auto-detect.
    For Hiligaynon-only meetings, ``prefer_forced`` / ``forced`` still map
    through ``_forced_language`` (``hil`` → ``tl``).
    """
    mode = (settings.whisper_final_language_mode or "auto").strip().lower()
    if mode in {"forced", "force", "tl"}:
        return _forced_language(requested)
    if mode in {"prefer_forced", "prefer-tl", "prefer_tl"}:
        return _forced_language(requested)
    # auto / detect / none
    return None


def _segment_time_coverage(segments: list[Segment], duration: float) -> float:
    """Fraction of ``duration`` covered by segment [start,end] intervals."""
    if duration <= 0 or not segments:
        return 0.0
    spans = sorted(
        (max(0.0, float(s.start)), max(0.0, float(s.end)))
        for s in segments
        if (s.text or "").strip()
    )
    if not spans:
        return 0.0
    covered = 0.0
    cur_a, cur_b = spans[0]
    for a, b in spans[1:]:
        if a <= cur_b:
            cur_b = max(cur_b, b)
        else:
            covered += max(0.0, cur_b - cur_a)
            cur_a, cur_b = a, b
    covered += max(0.0, cur_b - cur_a)
    return min(1.0, covered / duration)


def _largest_segment_gap(segments: list[Segment], duration: float) -> float:
    if duration <= 0:
        return 0.0
    points = sorted(
        (max(0.0, float(s.start)), max(0.0, float(s.end)))
        for s in segments
        if (s.text or "").strip()
    )
    if not points:
        return duration
    gap = points[0][0]  # leading silence/gap
    prev_end = points[0][1]
    for start, end in points[1:]:
        gap = max(gap, start - prev_end)
        prev_end = max(prev_end, end)
    gap = max(gap, duration - prev_end)
    return max(0.0, gap)


def _is_junk_transcript(text: str) -> bool:
    """True when Whisper output looks like silence/spam hallucination."""
    raw = (text or "").strip()
    cleaned = _collapse_hallucinations(raw)
    if not cleaned:
        return True
    # Mostly a repetition loop that collapsed away (e.g. Pag-papapapa… → Pag-pa).
    if len(raw) >= 24 and len(cleaned) < max(8, int(len(raw) * 0.3)):
        return True
    low = cleaned.lower()
    for phrase in _HALLUCINATION_PHRASES:
        if phrase in low:
            return True
    # Sentence repeated 3+ times (classic Whisper silence loop).
    parts = [p.strip() for p in re.split(r"[.!?]+", cleaned) if p.strip()]
    if len(parts) >= 3 and len(set(p.lower() for p in parts)) == 1:
        return True
    tokens = re.findall(r"[a-zA-ZÀ-ÿ0-9']+", cleaned)
    if not tokens:
        return True

    def _low_entropy_token(tok: str) -> bool:
        if len(tok) < 10:
            return False
        low = tok.lower()
        if len(set(low)) <= 3:
            return True
        for n in (1, 2, 3, 4):
            unit = low[:n]
            if not unit:
                continue
            repeats = len(low) // n
            if repeats >= 4 and low.startswith(unit * repeats):
                return True
        return False

    # Long nonsense tokens ("papapapapa…") often survive word-level filters.
    if any(_low_entropy_token(t) for t in tokens):
        return True
    # Dominant repeated token (e.g. pag-pag-pag… after collapse still short).
    if len(tokens) >= 4:
        counts: dict[str, int] = {}
        for t in tokens:
            key = t.lower()
            counts[key] = counts.get(key, 0) + 1
        top_n = max(counts.values())
        if top_n / len(tokens) >= 0.5:
            return True
    # Repeated bigram/trigram loops: "hindi ko hindi ko hindi ko…"
    # These freeze live captions after a couple of seconds if left unfiltered.
    if len(tokens) >= 6:
        norms = [t.lower() for t in tokens]
        for n in (2, 3):
            grams = [
                " ".join(norms[i : i + n]) for i in range(0, len(norms) - n + 1)
            ]
            if not grams:
                continue
            freq: dict[str, int] = {}
            for g in grams:
                freq[g] = freq.get(g, 0) + 1
            _top_g, top_c = max(freq.items(), key=lambda kv: kv[1])
            if top_c >= 3 and (top_c * n) >= int(len(norms) * 0.5):
                return True
    return False


def _energy_ok(pcm: np.ndarray, *, min_rms: float = 0.008) -> bool:
    """Skip near-silent windows that only produce Whisper hallucinations."""
    if pcm is None or len(pcm) == 0:
        return False
    rms = float(np.sqrt(np.mean(np.square(pcm.astype(np.float32)))))
    peak = float(np.max(np.abs(pcm)))
    return rms >= min_rms or peak >= 0.02


def transcribe_live(pcm: np.ndarray, language: str | None) -> list[Segment]:
    """Low-latency transcription of a live audio window.

    Uses overlapping windows upstream; decode uses ``task=transcribe`` and a
    configured language code (default ``tl``), with an auto-detect retry when
    forced-language output is empty/junk (common on English board speech).
    """
    if pcm is None or len(pcm) == 0:
        return []
    # Soften energy gate — clipped/quiet phrases were skipped entirely.
    if not _energy_ok(pcm, min_rms=0.004):
        return []

    cache = get_model_cache()
    model_id = live_model_id(language)
    model = cache.get_faster_whisper(model_id)
    lang = _forced_language(language)
    # Bias live decode toward Hiligaynon only when using a PH fine-tune; stock
    # Whisper often echoes long prompts into captions.
    live_prompt = (
        initial_prompt()
        if (settings.whisper_live_hiligaynon_model or "").strip()
        and is_philippine_language(language)
        else None
    )
    t0 = time.perf_counter()

    def _run(decode_language: str | None):
        with cache.fw_infer_lock(model_id):
            return model.transcribe(
                pcm,
                language=decode_language,
                task=_WHISPER_TASK,
                beam_size=3,
                best_of=3,
                temperature=0.0,
                # Softer live VAD — prior onset/offset dropped real speech.
                vad_filter=True,
                vad_parameters={
                    "onset": 0.30,
                    "offset": 0.20,
                    "min_speech_duration_ms": 80,
                    "min_silence_duration_ms": 350,
                    "speech_pad_ms": 300,
                },
                condition_on_previous_text=False,
                without_timestamps=False,
                no_speech_threshold=0.65,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                initial_prompt=live_prompt,
            )

    segments, info = _run(lang)
    out: list[Segment] = []
    for s in segments:
        text = _collapse_hallucinations((s.text or "").strip())
        if not text or not any(ch.isalnum() for ch in text):
            continue
        if _is_junk_transcript(text):
            continue
        out.append(Segment(text=text, start=s.start, end=s.end))

    # If forced-language decode produced only junk, retry once with auto-detect
    # so English / mixed board speech still captions.
    if not out:
        try:
            segments, info = _run(None)
            detected = getattr(info, "language", None)
            if detected:
                logger.info("Live auto-detect retry language=%s", detected)
            for s in segments:
                text = _collapse_hallucinations((s.text or "").strip())
                if text and any(ch.isalnum() for ch in text) and not _is_junk_transcript(text):
                    out.append(Segment(text=text, start=s.start, end=s.end))
        except Exception as exc:
            logger.debug("Live auto-detect retry failed: %s", exc)

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(
        "asr.live duration_ms=%d segments=%d pcm_samples=%d model=%s",
        elapsed_ms,
        len(out),
        int(len(pcm)),
        model_id,
    )
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


def _novel_suffix_from_window(
    previous_window: str,
    current_window: str,
    *,
    hop_fraction: float | None = None,
) -> str:
    """Return only the new words from an overlapping Whisper window.

    Strategy (kept intentionally simple for debuggability):
    1. Exact token suffix/prefix overlap vs the previous window.
    2. If that fails, keep the newest ``hop_fraction`` of the current window
       (default 0.5 for a 10s window / 5s hop) — never re-paste the whole window.
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
        # No reliable overlap — keep most of the window (not just the hop).
        # Using only ~50% previously dropped mid-window words when Whisper
        # rephrased the overlap enough to break exact token matching.
        frac = hop_fraction
        if frac is None:
            frac = 0.85
        frac = min(1.0, max(0.5, float(frac)))
        cut = max(1, int(round(len(cur_tokens) * frac)))
        novel = " ".join(cur_tokens[-cut:]).strip()
        logger.debug(
            "live.merge overlap=0 fallback_keep_tokens=%d/%d",
            cut,
            len(cur_tokens),
        )
        return novel

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

    Invariant: the returned caption never shrinks versus ``previous``.

    Merge path (simple, debuggable):
    1. Prefer window-to-window novel suffix (exact token overlap).
    2. Else novel vs caption tail only (never full multi-hour scan).
    3. Else hop-fraction / whole-window append with containment check.
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
        if text and not _is_junk_transcript(text):
            out.append(Segment(text=text, start=s.start, end=s.end))
    return out


def _audio_to_float32(audio_source) -> np.ndarray:
    if isinstance(audio_source, np.ndarray):
        return audio_source.astype(np.float32, copy=False)
    from . import audio as audio_svc

    return audio_svc.load_audio_float32(str(audio_source))


def _transcribe_final_hf(
    audio_source,
    language: str | None,
    model_id: str | None = None,
) -> list[Segment]:
    """Final pass with a Hugging Face fine-tuned Whisper checkpoint."""
    model_id = (model_id or hiligaynon_model_id()).strip()
    cache = get_model_cache()
    pipe = cache.get_hf_pipeline(model_id)
    samples = _audio_to_float32(audio_source)
    if samples.size == 0:
        return []

    lang = _final_decode_language(language)
    logger.info(
        "Final Whisper ASR with fine-tuned model '%s' (language=%s, task=%s)",
        model_id,
        lang,
        _WHISPER_TASK,
    )
    t0 = time.perf_counter()
    gen_kwargs = {"task": _WHISPER_TASK}
    if lang:
        gen_kwargs["language"] = lang
    with cache.hf_infer_lock(model_id):
        result = pipe(
            {"array": samples, "sampling_rate": int(settings.audio_sample_rate)},
            generate_kwargs=gen_kwargs,
            return_timestamps=True,
        )
    logger.info(
        "asr.final_hf duration_ms=%d samples=%d model=%s",
        int((time.perf_counter() - t0) * 1000),
        int(samples.size),
        model_id,
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


def _run_faster_whisper_final(
    model,
    model_id: str,
    audio_in,
    *,
    language: str | None,
    vad_filter: bool,
) -> tuple[list[Segment], object]:
    with get_model_cache().fw_infer_lock(model_id):
        segments, info = model.transcribe(
            audio_in,
            language=language,
            task=_WHISPER_TASK,
            beam_size=5,
            best_of=5,
            temperature=[0.0, 0.2, 0.4],
            vad_filter=vad_filter,
            vad_parameters=_FINAL_VAD_PARAMS if vad_filter else None,
            # Continuity across phrases — False caused more dropped clauses.
            condition_on_previous_text=True,
            without_timestamps=False,
            initial_prompt=initial_prompt(),
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
        )
    return _segments_to_list(segments), info


def _transcribe_final_faster_whisper(audio_source, language: str | None) -> list[Segment]:
    """Final pass via faster-whisper with coverage-aware retries.

    Root cause of missing spoken words on board meetings:
    forcing ``language=tl`` + aggressive VAD skipped long spans of real English
    speech (verified on production WAVs with continuous energy but empty
    transcript gaps of 20–30s). We now default to auto language and no VAD,
    and retry if timeline coverage is still poor.
    """
    model_id = final_faster_model_id(language)
    cache = get_model_cache()
    model = cache.get_faster_whisper(model_id)
    samples = (
        audio_source
        if isinstance(audio_source, np.ndarray)
        else _audio_to_float32(audio_source)
    )
    audio_in = samples
    duration = float(samples.size) / float(settings.audio_sample_rate) if isinstance(samples, np.ndarray) else 0.0
    if isinstance(audio_source, str):
        audio_in = audio_source
        # duration still from loaded samples above when ndarray; for path reload:
        if duration <= 0:
            samples = _audio_to_float32(audio_source)
            duration = float(samples.size) / float(settings.audio_sample_rate)

    mode = (settings.whisper_final_language_mode or "auto").strip().lower()
    primary_lang = _final_decode_language(language)
    use_vad = bool(settings.whisper_final_vad_filter)
    min_cov = min(0.95, max(0.2, float(settings.whisper_final_min_coverage)))

    attempts: list[tuple[str, str | None, bool]] = [
        ("primary", primary_lang, use_vad),
    ]
    # Coverage retries — order matters.
    if primary_lang is not None:
        attempts.append(("auto_retry", None, use_vad))
    if use_vad:
        attempts.append(("no_vad", primary_lang, False))
        attempts.append(("auto_no_vad", None, False))
    else:
        # Even with VAD off, a forced-language miss should retry auto / tl.
        if primary_lang is None:
            attempts.append(("forced_tl_retry", _forced_language(language), False))
        else:
            attempts.append(("auto_retry2", None, False))

    # Deduplicate attempt signatures.
    seen: set[tuple[str | None, bool]] = set()
    unique_attempts: list[tuple[str, str | None, bool]] = []
    for label, lang, vad in attempts:
        key = (lang, vad)
        if key in seen:
            continue
        seen.add(key)
        unique_attempts.append((label, lang, vad))

    best: list[Segment] = []
    best_score = (-1.0, -1, -1.0, 0.0)  # weighted words, words, cov, -gap
    t0 = time.perf_counter()
    for label, lang, vad in unique_attempts:
        try:
            segs, info = _run_faster_whisper_final(
                model, model_id, audio_in, language=lang, vad_filter=vad
            )
        except Exception as exc:
            logger.warning("Final ASR attempt %s failed: %s", label, exc)
            continue
        detected = getattr(info, "language", None)
        cov = _segment_time_coverage(segs, duration)
        words = len(" ".join(s.text for s in segs).split())
        gap = _largest_segment_gap(segs, duration)
        # Penalize long low-density spans ("we will talk" covering 30s).
        sparse_penalty = 0.0
        for s in segs:
            seg_dur = max(0.01, float(s.end) - float(s.start))
            seg_words = len((s.text or "").split())
            if seg_dur >= 8.0 and (seg_words / seg_dur) < 0.35:
                sparse_penalty += seg_dur
        density = words / max(duration, 1.0)
        logger.info(
            "asr.final_fw attempt=%s lang=%s vad=%s detected=%s "
            "coverage=%.2f gap=%.1fs words=%d density=%.2f sparse_pen=%.1f",
            label,
            lang,
            vad,
            detected,
            cov,
            gap,
            words,
            density,
            sparse_penalty,
        )
        # Word mass first (coverage-weighted), then raw words, then coverage.
        score = (words * cov - sparse_penalty, words, cov, -gap)
        if score > best_score:
            best_score = score
            best = segs
        # Early exit only when dense + well covered.
        if (
            cov >= max(min_cov, 0.75)
            and gap <= 10.0
            and sparse_penalty < 8.0
            and words >= max(12, int(duration * 0.8))
        ):
            break

    logger.info(
        "asr.final_fw duration_ms=%d model=%s best_score_words_x_cov=%.1f best_words=%d",
        int((time.perf_counter() - t0) * 1000),
        model_id,
        best_score[0] if best_score[0] >= 0 else 0.0,
        best_score[1] if best_score[1] >= 0 else 0,
    )
    return best


def _looks_like_hf_repo(model_id: str) -> bool:
    mid = (model_id or "").strip()
    return "/" in mid or mid.startswith(".") or mid.startswith("/")


def _transcribe_final_once(audio_source, language: str | None) -> list[Segment]:
    """Single-shot final ASR.

    ``WHISPER_FINAL_BACKEND=auto`` (default) tries Hiligaynon HF candidates in
    order (user fine-tune → Philippine fallback), then faster-whisper.
    """
    backend = resolve_final_backend(language)
    configured = (settings.whisper_final_backend or "auto").strip().lower()
    strict_hf = configured in {"huggingface", "hf", "transformers"}

    if backend == "huggingface":
        candidates = [
            mid for mid in hiligaynon_hf_candidates() if _looks_like_hf_repo(mid)
        ]
        for model_id in candidates:
            try:
                segs = _transcribe_final_hf(audio_source, language, model_id=model_id)
                joined = " ".join(s.text for s in segs)
                if segs and not _is_junk_transcript(joined):
                    logger.info(
                        "Final ASR using Hiligaynon HF model '%s' "
                        "(meeting_language=%s)",
                        model_id,
                        language,
                    )
                    return segs
                logger.warning(
                    "HF final ASR '%s' looked like hallucination; trying next candidate.",
                    model_id,
                )
            except TranscriptionUnavailable as exc:
                # Missing transformers/torch — keep trying other candidates is
                # pointless; fall back to FW unless the user forced HF-only.
                if strict_hf:
                    raise
                logger.warning(
                    "HF Whisper unavailable (%s); falling back to faster-whisper.",
                    exc,
                )
                break
            except Exception as exc:
                logger.exception(
                    "Hiligaynon Whisper '%s' failed (%s); trying next / FW fallback.",
                    model_id,
                    exc,
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
    Whisper model when enabled. Fall back to faster-whisper with auto language
    detection and coverage-aware retries (forced ``tl`` + aggressive VAD was
    dropping spoken English spans).

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
