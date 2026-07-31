"""Whisper transcription service — the ``TranscribeAudio`` integration.

Implements the two-pass pipeline:

* **Live pass** — fast Whisper (or an optional CTranslate2 Tagalog/Hiligaynon
  fine-tune) on overlapping 10s windows (5s hop). Tagalog (``tl``) uses
  Whisper's native ``tl`` token. Hiligaynon (``hil``) uses **auto-detect**
  plus a Hiligaynon prompt — Whisper has no ``hil`` token, and we do **not**
  force Tagalog decode for Ilonggo speech.
* **Final pass** — when ``WHISPER_FINAL_BACKEND=auto|huggingface``:
  Tagalog prefers custom → ``WHISPER_TAGALOG_MODEL`` → PH medium;
  Hiligaynon prefers custom fine-tune → ``WHISPER_HILIGAYNON_MODEL``
  (``rbcurzon/whisper-medium-ph``, Visayan-aware) then faster-whisper,
  with loudness normalization, short prompts, and auto-detect (no forced ``tl``).

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
        # Product defaults
        "hil",
        "hiligaynon",
        "ilonggo",
        "fil",
        "tl",
        "tagalog",
        "filipino",
        # UP-DSP Philippine Languages Database (PLD) codes
        "ceb",
        "cebuano",
        "bisaya",
        "ilo",
        "ilokano",
        "ilocano",
        "bik",
        "bikol",
        "bikolano",
        "war",
        "waray",
        "pam",
        "kapampangan",
        "pag",
        "pangasinense",
        "pangasinan",
        "tsg",
        "tausug",
    }
)
_TAGALOG_LANGUAGE_LABELS = frozenset({"tl", "tagalog", "fil", "filipino"})
_HILIGAYNON_LANGUAGE_LABELS = frozenset({"hil", "hiligaynon", "ilonggo"})

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
    "i love this thing",
    "i don't know what that",
)


class TranscriptionUnavailable(RuntimeError):
    """Raised when the Whisper backend cannot be loaded."""


@dataclass
class Segment:
    text: str
    start: float
    end: float
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    low_confidence: bool = False


@dataclass
class LanguageDetection:
    """Whisper language-detect metadata for analysis / debugging.

    ``detected_by`` is ``whisper`` when auto-detect chose the code, or
    ``forced_fallback`` when a forced decode code (e.g. Tagalog ``tl``) was used.
    """

    language: str | None = None
    confidence: float | None = None
    detected_by: str = "whisper"

    def as_dict(self) -> dict:
        return {
            "language": self.language,
            "confidence": self.confidence,
            "detected_by": self.detected_by,
        }


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


def _dedupe_model_ids(ordered: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for mid in ordered:
        mid = (mid or "").strip()
        if not mid or mid in seen:
            continue
        seen.add(mid)
        out.append(mid)
    return out


def is_auto_language(language: str | None) -> bool:
    """True when Whisper should detect language (no user Spoken language pick)."""
    lang = (language or "").strip().lower()
    # Unset / empty always means detect — do not inherit a stale .env hil/tl.
    if not lang:
        return True
    return lang in {"auto", "detect", "none"}


def is_tagalog_language(language: str | None) -> bool:
    lang = (language or "").strip().lower()
    return lang in _TAGALOG_LANGUAGE_LABELS


def is_hiligaynon_language(language: str | None) -> bool:
    lang = (language or "").strip().lower()
    return lang in _HILIGAYNON_LANGUAGE_LABELS


def whisper_language_arg(language: str | None) -> str | None:
    """Map an app/UI language label to a Whisper-native decode code.

    Whisper understands ``tl`` (Tagalog) and ``en``, but **not** ``fil``, ``hil``,
    or ``ceb``. Passing ``fil`` silently breaks Tagalog forcing; passing ``hil``
    is invalid. Hiligaynon / Cebuano / auto → ``None`` (auto-detect + prompt).
    """
    raw = (language or "").strip().lower()
    if not raw or raw in {"auto", "detect", "none"}:
        return None
    if raw in {"en", "english"}:
        return "en"
    if is_tagalog_language(raw):
        return "tl"
    if is_hiligaynon_language(raw) or raw in {
        "ceb",
        "cebuano",
        "bisaya",
        "war",
        "waray",
    }:
        return None
    # Already a short Whisper ISO code (e.g. session lock stored ``tl``/``en``).
    if len(raw) <= 3 and raw.isalpha() and raw not in {"fil", "hil", "ceb"}:
        return raw
    return None


def normalize_meeting_language(language: str | None) -> str:
    """Normalize API/UI language aliases to stable meeting labels."""
    lang = (language or "").strip().lower() or "auto"
    if lang in {"detect", "none"}:
        return "auto"
    if lang in {"hiligaynon", "ilonggo"}:
        return "hil"
    if lang in {"tagalog", "fil", "filipino"}:
        return "tl"
    if lang == "english":
        return "en"
    return lang


def effective_asr_language(language: str | None) -> str:
    """Resolve meeting/UI language to the ASR bias used at runtime.

    ``auto`` / empty maps to ``whisper_default_language`` (Hiligaynon by default)
    so Iloilo board meetings get Hiligaynon prompts + PH models without a
    manual Spoken language selection.
    """
    lang = (language or "").strip().lower()
    if is_auto_language(lang) or not lang:
        default = (settings.whisper_default_language or "hil").strip().lower()
        return default or "hil"
    return lang


def is_philippine_language(language: str | None) -> bool:
    """True when ASR should use the Philippine model path.

    ``auto`` counts as PH-oriented (board meetings in hil/tl/fil/en mix).
    """
    if is_auto_language(language):
        return True
    lang = (language or "").strip().lower()
    return lang in _PH_LANGUAGE_LABELS


def hiligaynon_hf_candidates() -> list[str]:
    """Ordered HF/local Whisper ids for Hiligaynon final ASR.

    1. Custom Hiligaynon fine-tune (best when available)
    2. Philippine dialect medium (``rbcurzon/whisper-medium-ph``) — trained on
       Visayan/PH speech and recovers Ilonggo forms (``gid``, ``nakapoy``,
       ``mangita sang``) far better than stock Whisper+``tl``

    Quiet-mic hallucinations are mitigated by ``amplify_for_asr`` before decode;
    results are still scored against faster-whisper and the better transcript wins.
    """
    return _dedupe_model_ids(
        [
            settings.whisper_hiligaynon_fine_tuned_model,
            settings.whisper_hiligaynon_model,
        ]
    )


def tagalog_hf_candidates() -> list[str]:
    """Ordered HF/local Whisper ids for Tagalog / Filipino final ASR.

    1. User Tagalog fine-tune
    2. Tagalog-specific HF model (default ``LWobole/whisper-small-tagalog``)
    3. Broader Philippine medium model (``rbcurzon/whisper-medium-ph``)
    """
    return _dedupe_model_ids(
        [
            settings.whisper_tagalog_fine_tuned_model,
            settings.whisper_tagalog_model,
            settings.whisper_hiligaynon_model,
        ]
    )


def auto_hf_candidates() -> list[str]:
    """HF candidates when language is unknown / auto-detected.

    Prefer Philippine medium. Tagalog-small is omitted here — it often wins
    first-wins/scoring with high-coverage **repetition loops** on mixed speech.
    Explicit Tagalog meetings still use ``tagalog_hf_candidates()``.
    """
    return _dedupe_model_ids(
        [
            settings.whisper_hiligaynon_fine_tuned_model,
            settings.whisper_tagalog_fine_tuned_model,
            settings.whisper_hiligaynon_model,
        ]
    )


def philippine_hf_candidates(language: str | None) -> list[str]:
    """HF candidates for the meeting language (auto / Tagalog / Hiligaynon).

    Non-Philippine languages (e.g. English) return ``[]`` so final ASR uses
    faster-whisper only — PH-medium is a poor fit for English-majority audio.
    """
    raw = (language or "").strip().lower()
    # Honor explicit English before effective_asr_language remaps unknowns.
    if raw in {"en", "english"}:
        return []
    lang = effective_asr_language(language)
    if is_tagalog_language(lang):
        return tagalog_hf_candidates()
    if is_hiligaynon_language(lang):
        return hiligaynon_hf_candidates()
    if is_philippine_language(lang) or is_auto_language(raw):
        return auto_hf_candidates()
    return []


def hiligaynon_model_id() -> str:
    """Primary Hugging Face / local id for the Hiligaynon final pass."""
    cands = hiligaynon_hf_candidates()
    if cands:
        return cands[0]
    return (settings.whisper_final_model or "medium").strip()


def initial_prompt(
    language: str | None = None,
    *,
    extra_terms: list[str] | None = None,
) -> str | None:
    """Language-aware short prompt (avoid long prompts — Whisper may echo them).

    ``extra_terms`` are optional meeting proper nouns (attendees / org terms)
    appended as a short custom vocabulary hint.
    """
    lang = effective_asr_language(language)
    if is_hiligaynon_language(lang):
        prompt = (settings.whisper_hiligaynon_initial_prompt or "").strip()
        if not prompt:
            prompt = (settings.whisper_initial_prompt or "").strip()
    elif is_tagalog_language(lang):
        prompt = (settings.whisper_tagalog_initial_prompt or "").strip()
        if not prompt:
            prompt = (settings.whisper_initial_prompt or "").strip()
    else:
        prompt = (settings.whisper_initial_prompt or "").strip()
    terms = [t.strip() for t in (extra_terms or []) if isinstance(t, str) and t.strip()]
    # Keep short — Whisper echoes long prompts.
    terms = terms[:24]
    if terms:
        vocab = ", ".join(terms)
        hint = f" Vocabulary: {vocab}."
        prompt = f"{prompt}{hint}".strip() if prompt else f"Vocabulary: {vocab}."
    return prompt or None


def parse_custom_vocab(raw) -> list[str]:
    """Parse meeting custom_vocab (JSON list, newlines, or commas)."""
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            import json

            parsed = json.loads(text)
            items = parsed if isinstance(parsed, list) else text.splitlines()
        except Exception:
            items = text.splitlines() if "\n" in text else [p for p in text.split(",")]
    else:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, str):
            continue
        term = item.strip()
        if not term:
            continue
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(term)
    return out[:24]


def _segment_from_whisper(s, *, text: str) -> Segment | None:
    """Build a Segment from a faster-whisper segment, applying confidence gates."""
    if not text or not any(ch.isalnum() for ch in text):
        return None
    if _is_junk_transcript(text):
        return None
    avg_lp = getattr(s, "avg_logprob", None)
    no_sp = getattr(s, "no_speech_prob", None)
    try:
        avg_lp_f = float(avg_lp) if avg_lp is not None else None
    except (TypeError, ValueError):
        avg_lp_f = None
    try:
        no_sp_f = float(no_sp) if no_sp is not None else None
    except (TypeError, ValueError):
        no_sp_f = None

    # Hard drop: almost certainly silence hallucination.
    # Whisper often marks quiet-but-real speech with high no_speech_prob; keep
    # segments that still carry substantial lexical content and flag them instead.
    low = False
    if no_sp_f is not None and no_sp_f >= float(settings.asr_max_no_speech_prob):
        word_count = len(text.split())
        alpha_chars = sum(1 for ch in text if ch.isalpha())
        if word_count >= 4 and alpha_chars >= 12:
            logger.info(
                "asr.keep_low_conf no_speech_prob=%.3f words=%d text=%r",
                no_sp_f,
                word_count,
                text[:80],
            )
            low = True
        else:
            logger.info(
                "asr.drop_segment no_speech_prob=%.3f text=%r",
                no_sp_f,
                text[:80],
            )
            return None
    if avg_lp_f is not None and avg_lp_f <= float(settings.asr_min_avg_logprob) - 0.5:
        # Far below threshold — drop.
        logger.info("asr.drop_segment avg_logprob=%.3f text=%r", avg_lp_f, text[:80])
        return None

    if avg_lp_f is not None and avg_lp_f < float(settings.asr_flag_avg_logprob):
        low = True
    if no_sp_f is not None and no_sp_f >= float(settings.asr_flag_no_speech_prob):
        low = True

    return Segment(
        text=text,
        start=float(getattr(s, "start", 0.0) or 0.0),
        end=float(getattr(s, "end", 0.0) or 0.0),
        avg_logprob=avg_lp_f,
        no_speech_prob=no_sp_f,
        low_confidence=low,
    )


def live_model_id(language: str | None) -> str:
    """faster-whisper model for live captions (optional PH CT2 fine-tune)."""
    lang = effective_asr_language(language)
    if is_tagalog_language(lang):
        custom = (settings.whisper_live_tagalog_model or "").strip()
        if custom:
            return custom
    if is_hiligaynon_language(lang):
        custom = (settings.whisper_live_hiligaynon_model or "").strip()
        if custom:
            return custom
    if is_philippine_language(lang):
        custom = (settings.whisper_live_hiligaynon_model or "").strip()
        if custom:
            return custom
    return (settings.whisper_live_model or "small").strip()


def final_faster_model_id(language: str | None) -> str:
    """faster-whisper model for final fallback / FW-only backend.

    Always use the configured final size (default ``medium``). Live CT2
    fine-tunes are for captions only — using them here downgraded accuracy.
    """
    return (settings.whisper_final_model or "medium").strip()


def resolve_final_backend(language: str | None) -> str:
    """Normalize final ASR backend selection.

    ``auto`` prefers HF Tagalog/Hiligaynon/PH candidates for PH/auto meetings,
    otherwise faster-whisper.
    """
    backend = (settings.whisper_final_backend or "auto").strip().lower()
    if backend in {"huggingface", "hf", "transformers"}:
        return "huggingface"
    if backend in {"faster-whisper", "fw", "ctranslate2", "ct2"}:
        return "faster-whisper"
    # auto / default
    if philippine_hf_candidates(language):
        return "huggingface"
    return "faster-whisper"


def _normalize_detected_language(code: str | None) -> str | None:
    lang = (code or "").strip().lower()
    if not lang or lang in {"auto", "detect", "none"}:
        return None
    return lang


def _clamp_confidence(value: object) -> float | None:
    if value is None:
        return None
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return None
    if conf != conf:  # NaN
        return None
    return max(0.0, min(1.0, conf))


def _language_detection_from_info(
    info: object | None,
    *,
    decode_language: str | None,
    fallback_language: str | None = None,
) -> LanguageDetection:
    """Build detection metadata from a faster-whisper ``TranscriptionInfo``.

    ``decode_language=None`` means Whisper auto-detect was used → ``whisper``.
    A non-null decode code means we forced the token → ``forced_fallback``.
    """
    detected = _normalize_detected_language(
        getattr(info, "language", None) if info is not None else None
    )
    confidence = _clamp_confidence(
        getattr(info, "language_probability", None) if info is not None else None
    )
    forced = _normalize_detected_language(decode_language)
    if forced is None:
        return LanguageDetection(
            language=detected or _normalize_detected_language(fallback_language),
            confidence=confidence,
            detected_by="whisper",
        )
    return LanguageDetection(
        language=detected or forced or _normalize_detected_language(fallback_language),
        confidence=confidence,
        detected_by="forced_fallback",
    )


def _language_detection_forced(
    language: str | None,
    *,
    confidence: float | None = None,
) -> LanguageDetection:
    return LanguageDetection(
        language=_normalize_detected_language(language),
        confidence=_clamp_confidence(confidence),
        detected_by="forced_fallback",
    )


_REPEAT_WORD_RE = re.compile(r"\b(\w+)(?:\s+\1){3,}\b", re.IGNORECASE)
_STUTTER_RE = re.compile(r"([A-Za-zÀ-ÿ])(?:-\1){3,}", re.IGNORECASE)
_HYPHEN_LOOP_RE = re.compile(r"\b(\w+)(?:-\1){2,}\b", re.IGNORECASE)
# "papapapapa" / "nanananana" inside a single token.
_SYLLABLE_LOOP_RE = re.compile(r"\b([A-Za-zÀ-ÿ]{1,4}?)\1{3,}\b", re.IGNORECASE)
# Multi-word loops: "ang kanil ang kanil ang kanil…" / "I don't know why…"
_PHRASE_LOOP_RE = re.compile(
    r"\b((?:\w+'?\w*\s+){0,5}\w+'?\w*)(?:\s+\1){2,}\b",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _collapse_repeated_sentences(text: str) -> str:
    """Keep the first copy of consecutive identical sentences."""
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]
    if len(parts) < 2:
        return text.strip()
    out: list[str] = []
    prev_norm = ""
    streak = 0
    for part in parts:
        norm = re.sub(r"\s+", " ", part.lower())
        if norm == prev_norm:
            streak += 1
            if streak >= 1:
                continue
        else:
            streak = 0
            prev_norm = norm
            out.append(part)
    if not out:
        return text.strip()
    # Preserve terminal punctuation when present on the last kept sentence.
    joined = ". ".join(out)
    if text.rstrip().endswith((".", "!", "?")) and not joined.endswith((".", "!", "?")):
        joined += "."
    return joined


def _collapse_hallucinations(text: str) -> str:
    """Collapse obvious Whisper repetition loops (words, phrases, sentences)."""
    prev = None
    text = (text or "").strip()
    while prev != text:
        prev = text
        text = _REPEAT_WORD_RE.sub(r"\1", text)
        text = _STUTTER_RE.sub(r"\1", text)
        text = _HYPHEN_LOOP_RE.sub(r"\1", text)
        text = _SYLLABLE_LOOP_RE.sub(r"\1", text)
        text = _PHRASE_LOOP_RE.sub(r"\1", text)
        text = re.sub(r"\s+", " ", text).strip()
    return _collapse_repeated_sentences(text)


def _forced_language(requested: str | None = None) -> str | None:
    """Whisper language code when forcing decode.

    Only English and Tagalog/Filipino have Whisper-native codes we force.
    Hiligaynon is **never** mapped to ``tl`` — callers must use auto-detect
    (``None``) plus the Hiligaynon prompt / PH model instead.
    Always returns Whisper-native codes (``tl`` / ``en``), never ``fil``.
    """
    raw = (requested or "").strip().lower()
    # Honor explicit labels before auto→hil remapping.
    if raw in {"en", "english"} or is_tagalog_language(raw):
        return whisper_language_arg(raw)
    app_lang = effective_asr_language(requested)
    if app_lang in {"en", "english"} or is_tagalog_language(app_lang):
        return whisper_language_arg(app_lang)
    return None

def _final_language_mode(requested: str | None) -> str:
    """Resolve final language mode.

    ``auto`` resolves to ``whisper_default_language`` (Hiligaynon by default),
    then applies that language's mode. Hiligaynon defaults to auto-detect
    (no forced Tagalog token).
    """
    lang = effective_asr_language(requested)
    if is_tagalog_language(lang):
        return (
            settings.whisper_tagalog_final_language_mode
            or settings.whisper_final_language_mode
            or "prefer_forced"
        ).strip().lower()
    if is_hiligaynon_language(lang):
        return (
            settings.whisper_hiligaynon_final_language_mode
            or settings.whisper_final_language_mode
            or "auto"
        ).strip().lower()
    return (settings.whisper_final_language_mode or "prefer_forced").strip().lower()


def _final_decode_language(requested: str | None) -> str | None:
    """Language for the final pass.

    Hiligaynon / auto→hil: always ``None`` (Whisper auto-detect).
    Explicit Tagalog may force ``tl`` when ``prefer_forced`` is configured.
    """
    lang = effective_asr_language(requested)
    if is_hiligaynon_language(lang):
        return None
    mode = _final_language_mode(requested)
    if mode in {"forced", "force", "tl", "prefer_forced", "prefer-tl", "prefer_tl"}:
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
    # Soft-speech PH hallucinations often pad with "?..." / "......" glue.
    # Skip this check when the text still looks like real Visayan/Hiligaynon,
    # unless ellipsis density is extreme (PH-medium quiet-mic spam).
    ellipsis_hits = (
        cleaned.count("?...")
        + cleaned.count("......")
        + cleaned.count("?... ")
        + len(re.findall(r"\.{3,}", cleaned))
    )
    extreme_ellipsis = ellipsis_hits >= 12 and ellipsis_hits >= int(len(tokens) * 0.35)
    mild_ellipsis = (
        ellipsis_hits >= 6
        and ellipsis_hits >= max(3, int(len(tokens) * 0.06))
        and _visayan_marker_ratio(cleaned) < 0.04
    )
    if extreme_ellipsis or mild_ellipsis:
        return True
    # Glued punctuation without spaces ("world!ang") is almost never real speech.
    glued = len(re.findall(r"[A-Za-zÀ-ÿ][!?.,;:][A-Za-zÀ-ÿ]", cleaned))
    if glued >= 2 and _visayan_marker_ratio(cleaned) < 0.06:
        return True
    if glued >= 1 and len(tokens) >= 6 and _visayan_marker_ratio(cleaned) < 0.06:
        adj_dups = sum(
            1
            for i in range(len(tokens) - 1)
            if tokens[i].lower() == tokens[i + 1].lower() and len(tokens[i]) >= 3
        )
        if adj_dups >= 1:
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


def transcribe_live(
    pcm: np.ndarray,
    language: str | None,
    *,
    extra_terms: list[str] | None = None,
) -> tuple[list[Segment], LanguageDetection | None]:
    """Low-latency transcription of a live audio window.

    When ``WHISPER_LIVE_BACKEND=auto|rnnt`` and NeMo is installed, Tagalog
    meetings may use FastConformer RNN-T for live captions. Hiligaynon-biased
    ``auto`` stays on Whisper (Tagalog RNNT maps Ilonggo poorly). On RNNT
    failure, always falls back to faster-whisper.

    Returns ``(segments, language_detection)``.
    """
    if pcm is None or len(pcm) == 0:
        return [], None
    from . import audio as audio_svc
    from . import pipeline_metrics, vad as vad_svc

    raw = pcm.astype(np.float32, copy=False)
    sr = int(settings.audio_sample_rate or 16000)
    # Tier 1: VAD gate before Whisper (biggest hallucination source = silence).
    vad_result = vad_svc.detect_speech(raw, sample_rate=sr, live=True)
    if not vad_result.has_speech:
        logger.info(
            "asr.live vad_skip backend=%s %s samples=%d",
            vad_result.backend,
            vad_result.reason,
            len(raw),
        )
        return [], None
    # Gate on *raw* PCM before AGC — amplifying silence first made every
    # quiet tail look like speech and Whisper re-looped the last phrase.
    if not _energy_ok(raw, min_rms=0.006):
        return [], None
    # Full live windows whose newest hop is silence: user stopped talking.
    # Skip so Whisper cannot re-decode old speech from a silent tail.
    window_s = float(settings.whisper_live_window_seconds or 10.0)
    hop_s = float(settings.whisper_live_hop_seconds or 5.0)
    if len(raw) >= int(0.85 * window_s * sr):
        hop_n = max(1, int(hop_s * sr))
        if len(raw) >= hop_n and not _energy_ok(raw[-hop_n:], min_rms=0.005):
            logger.info("asr.live skip_silent_tail samples=%d hop=%d", len(raw), hop_n)
            return [], None

    # Mild numpy AGC only — never dynaudnorm on live (boosts room noise).
    pcm = audio_svc.amplify_for_asr(
        raw,
        target_rms=0.06,
        max_gain=4.0,
        allow_dynaudnorm=False,
    )
    if not _energy_ok(pcm, min_rms=0.008):
        return [], None

    # Optional NeMo RNN-T live path (Tagalog meetings when enabled).
    try:
        from . import rnnt as rnnt_svc

        if rnnt_svc.should_use_rnnt_live(language) and rnnt_svc.is_available():
            try:
                segs, detection = rnnt_svc.transcribe_live(pcm, language)
                cleaned: list[Segment] = []
                for s in segs:
                    text = _collapse_hallucinations((s.text or "").strip())
                    text = _strip_initial_prompt_echo(
                        text, initial_prompt(language, extra_terms=extra_terms)
                    )
                    if (
                        text
                        and any(ch.isalnum() for ch in text)
                        and not _is_junk_transcript(text)
                    ):
                        cleaned.append(
                            Segment(
                                text=text,
                                start=s.start,
                                end=s.end,
                                avg_logprob=getattr(s, "avg_logprob", None),
                                no_speech_prob=getattr(s, "no_speech_prob", None),
                                low_confidence=bool(
                                    getattr(s, "low_confidence", False)
                                ),
                            )
                        )
                if cleaned:
                    return cleaned, detection
                logger.info("RNNT live returned empty/junk; falling back to Whisper")
            except Exception as exc:
                logger.warning("RNNT live failed (%s); falling back to Whisper", exc)
    except Exception as exc:
        logger.debug("RNNT module unavailable: %s", exc)

    cache = get_model_cache()
    model_id = live_model_id(language)
    model = cache.get_faster_whisper(model_id)
    # Hiligaynon / auto→hil: Whisper auto-detect (never force Tagalog ``tl``).
    # Explicit Tagalog still uses native ``tl``; English uses ``en``.
    # When language is already locked to a concrete code (session lock), prefer it.
    raw_label = (language or "").strip().lower()
    effective = effective_asr_language(language)
    if raw_label in {"en", "english"}:
        primary_lang: str | None = "en"
    elif is_tagalog_language(raw_label) or (
        is_tagalog_language(effective) and not is_auto_language(raw_label)
    ):
        primary_lang = "tl"
    elif is_auto_language(raw_label) or is_hiligaynon_language(raw_label) or is_hiligaynon_language(effective):
        # Hiligaynon-biased auto: never force tl. Locked concrete non-tl/en codes
        # still use auto-detect (Whisper has no hil token).
        primary_lang = None
    elif raw_label:
        # Session lock may pass a Whisper ISO code (e.g. "en", "tl").
        # Never forward ``fil`` / ``hil`` — Whisper does not understand them.
        primary_lang = whisper_language_arg(raw_label)
    else:
        primary_lang = None

    # Always bias live PH/Hiligaynon decode with the short language prompt + vocab.
    live_prompt = initial_prompt(language, extra_terms=extra_terms)
    t0 = time.perf_counter()
    winning_decode: str | None = primary_lang
    winning_info: object | None = None

    def _run(decode_language: str | None):
        # Never forward ``fil`` / ``hil`` — Whisper only accepts native codes.
        lang_arg = whisper_language_arg(decode_language) if decode_language else None
        with cache.fw_infer_lock(model_id):
            return model.transcribe(
                pcm,
                language=lang_arg,
                task=_WHISPER_TASK,
                beam_size=5,
                best_of=5,
                temperature=0.0,
                # Soft VAD drops silent tails that caused phrase loops.
                vad_filter=True,
                condition_on_previous_text=False,
                without_timestamps=False,
                # Stricter than before — silence was emitting looped captions.
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.2,
                log_prob_threshold=-1.0,
                initial_prompt=live_prompt,
            )

    with pipeline_metrics.track("asr.live"):
        segments, info = _run(primary_lang)
    winning_info = info
    out: list[Segment] = []
    for s in segments:
        text = _collapse_hallucinations((s.text or "").strip())
        text = _strip_initial_prompt_echo(text, live_prompt)
        seg = _segment_from_whisper(s, text=text)
        if seg:
            out.append(seg)

    # Empty/junk: Tagalog can retry auto↔tl. Hiligaynon stays on auto only.
    if not out and is_tagalog_language(effective):
        retry_lang = None if primary_lang is not None else "tl"
        if retry_lang != primary_lang:
            try:
                segments, info = _run(retry_lang)
                winning_decode = retry_lang
                winning_info = info
                for s in segments:
                    text = _collapse_hallucinations((s.text or "").strip())
                    text = _strip_initial_prompt_echo(text, live_prompt)
                    seg = _segment_from_whisper(s, text=text)
                    if seg:
                        out.append(seg)
            except Exception as exc:
                logger.debug("Live decode retry failed: %s", exc)

    detection = _language_detection_from_info(
        winning_info, decode_language=winning_decode
    )
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(
        "asr.live duration_ms=%d segments=%d pcm_samples=%d model=%s "
        "detected=%s confidence=%s by=%s low_conf=%d",
        elapsed_ms,
        len(out),
        int(len(pcm)),
        model_id,
        detection.language,
        detection.confidence,
        detection.detected_by,
        sum(1 for s in out if s.low_confidence),
    )
    return out, detection


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
        # No reliable overlap — keep only the newest hop fraction.
        # Keeping ~85% re-appended hallucinated repeats after the user stopped.
        frac = hop_fraction
        if frac is None:
            frac = 0.5
        frac = min(0.75, max(0.35, float(frac)))
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


def _near_duplicate_window(previous_window: str, current_window: str) -> bool:
    """True when the new window mostly repeats the previous one (silence loop)."""
    a = _norm_tokens(previous_window)
    b = _norm_tokens(current_window)
    if not b:
        return True
    if not a:
        return False
    if a == b:
        return True
    sb = set(b)
    if not sb:
        return True
    inter = len(set(a) & sb)
    # ≥85% of current tokens already in previous → treat as re-emit / loop.
    return inter / len(sb) >= 0.85 and len(b) <= len(a) + 3


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

    # Silence / hallucination often re-emits the last window — do not grow.
    if previous_window and _near_duplicate_window(previous_window, cur):
        return prev

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
        seg = _segment_from_whisper(s, text=text)
        if seg:
            out.append(seg)
    return out


def _audio_to_float32(audio_source) -> np.ndarray:
    if isinstance(audio_source, np.ndarray):
        return audio_source.astype(np.float32, copy=False)
    from . import audio as audio_svc

    return audio_svc.load_audio_float32(str(audio_source))


def _prepare_asr_audio(audio_source) -> np.ndarray:
    """Load float32 PCM and amplify quiet captures for Whisper."""
    from . import audio as audio_svc

    samples = _audio_to_float32(audio_source)
    if samples.size == 0:
        return samples
    return audio_svc.amplify_for_asr(samples)


def _parse_hf_asr_result(result, samples: np.ndarray) -> list[Segment]:
    """Normalize Hugging Face ASR pipeline output into ``Segment`` rows."""
    duration = float(samples.size) / float(settings.audio_sample_rate or 16000)
    chunks = result.get("chunks") if isinstance(result, dict) else None
    if chunks:
        out: list[Segment] = []
        for chunk in chunks:
            text = _collapse_hallucinations((chunk.get("text") or "").strip())
            if not text:
                continue
            ts = chunk.get("timestamp") or (None, None)
            start = float(ts[0] or 0.0)
            end_raw = ts[1] if len(ts) > 1 else None
            end = float(end_raw) if end_raw is not None else max(start, duration)
            if end < start:
                end = start
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
    return [Segment(text=text, start=0.0, end=duration)]


def _hf_prompt_ids(pipe, prompt: str | None):
    """Encode an optional initial prompt for Whisper ``generate``."""
    text = (prompt or "").strip()
    if not text:
        return None
    tokenizer = getattr(pipe, "tokenizer", None)
    if tokenizer is None:
        return None
    try:
        encoded = tokenizer(text, return_tensors=None, add_special_tokens=False)
        ids = encoded.get("input_ids") if isinstance(encoded, dict) else encoded
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        if ids:
            return ids
    except Exception as exc:
        logger.debug("Could not encode HF Whisper prompt: %s", exc)
    return None


def _score_segments(segments: list[Segment], duration: float) -> float:
    words = sum(len((s.text or "").split()) for s in segments)
    cov = _segment_time_coverage(segments, duration)
    return float(words) * max(0.15, cov)


def _strip_initial_prompt_echo(text: str, prompt: str | None) -> str:
    """Remove Whisper's habit of pasting initial_prompt tokens into output."""
    raw = (text or "").strip()
    tip = (prompt or "").strip()
    if not raw or not tip:
        return raw
    # Drop an exact leading copy of the prompt.
    if raw.lower().startswith(tip.lower()):
        raw = raw[len(tip) :].lstrip(" .,-:;|")
    # Isolated prompt content words (e.g. "Sang... Nga... Mga...") leaking mid-text.
    prompt_tokens = {
        t.lower()
        for t in re.findall(r"[A-Za-zÀ-ÿ']+", tip)
        if len(t) >= 3
    }
    # Never strip ubiquitous PH particles that also appear in real speech.
    keep = {
        "ang",
        "mga",
        "nga",
        "sang",
        "kag",
        "sa",
        "ko",
        "mo",
        "ni",
        "na",
        "pa",
        "lang",
        "man",
        "kay",
        "kon",
        "indi",
        "dili",
        "wala",
        "english",
        "hiligaynon",
        "ilonggo",
        "board",
        "meeting",
        "diskusyon",
        "discussion",
    }
    drop = prompt_tokens - keep
    if not drop:
        return raw
    parts = re.split(r"(\s+)", raw)
    out: list[str] = []
    for part in parts:
        core = re.sub(r"^[^A-Za-zÀ-ÿ0-9']+|[^A-Za-zÀ-ÿ0-9']+$", "", part)
        if core.lower() in drop and not re.search(r"[.!?]", part):
            # Skip bare echoed prompt tokens (keep if part of a sentence fragment).
            continue
        out.append(part)
    cleaned = "".join(out)
    cleaned = re.sub(r"(\s*\.\.\.\s*){2,}", "... ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" .")
    return cleaned or raw


_VISAYAN_MARKERS = frozenset(
    {
        "gid",
        "guid",
        "indi",
        "dili",
        "buwas",
        "buas",
        "nakapoy",
        "mangita",
        "kabalo",
        "subong",
        "subung",
        "sige",
        "dayon",
        "ila",
        "inyo",
        "naton",
        "ninyo",
        "kamo",
        "ako",
        "siya",
        "nila",
        "kon",
        "kung",
        "pero",
        "tapos",
        "wala",
        "may",
        "ara",
        "diri",
        "dida",
        "didto",
    }
)


def _visayan_marker_ratio(text: str) -> float:
    tokens = [t.lower() for t in re.findall(r"[A-Za-zÀ-ÿ']+", text or "")]
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in _VISAYAN_MARKERS)
    return hits / float(len(tokens))


def _garble_penalty(text: str) -> float:
    """Penalty multiplier (≤1) for PH-medium style garble that still passes junk filters.

    Catches glued punctuation (``world!ang``), adjacent duplicate tokens
    (``kaysa kaysa``), and hyphenated fragment spam (``g-morong``).
    """
    raw = (text or "").strip()
    if not raw:
        return 0.0
    penalty = 1.0
    glued = len(re.findall(r"[A-Za-zÀ-ÿ][!?.,;:][A-Za-zÀ-ÿ]", raw))
    if glued:
        # One glued hit is suspicious; two+ usually means broken tokenization.
        penalty *= max(0.12, 1.0 - 0.45 * glued)
    tokens = [t.lower() for t in re.findall(r"[A-Za-zÀ-ÿ0-9']+", raw)]
    if len(tokens) >= 4:
        adj_dups = sum(
            1
            for i in range(len(tokens) - 1)
            if tokens[i] == tokens[i + 1] and len(tokens[i]) >= 3
        )
        if adj_dups:
            penalty *= max(0.2, 1.0 - 0.28 * adj_dups)
    # Hyphenated nonsense fragments common in broken HF PH output.
    hyphen_frags = len(re.findall(r"\b[A-Za-zÀ-ÿ]{1,3}-[A-Za-zÀ-ÿ]{3,}\b", raw))
    if hyphen_frags >= 1:
        penalty *= max(0.25, 1.0 - 0.3 * hyphen_frags)
    # Dense missing-space globs: letters stuck across sentence boundaries.
    no_space_runs = len(re.findall(r"[a-z]{2,}[A-Z][a-z]{2,}", raw))
    if no_space_runs:
        penalty *= max(0.3, 1.0 - 0.25 * no_space_runs)
    # Phonetic mush: many short rare tokens with almost no function words.
    if len(tokens) >= 12:
        short_odd = sum(1 for t in tokens if 4 <= len(t) <= 8 and t.endswith(("i", "o", "a")))
        if short_odd >= max(6, int(len(tokens) * 0.35)) and _visayan_marker_ratio(raw) < 0.08:
            penalty *= 0.45
    return penalty


def _candidate_quality_score(segments: list[Segment], duration: float) -> float:
    """Rank ASR candidates by coverage, word mass, and lexical diversity.

    Collapses Whisper repetition loops first so a hallucinating model cannot
    beat a cleaner transcript purely by word count. Boosts transcripts that
    retain Visayan/Hiligaynon markers (``gid``, ``nakapoy``, ``indi``…).
    """
    if not segments:
        return -1.0
    raw = " ".join((s.text or "").strip() for s in segments if (s.text or "").strip())
    if not raw:
        return -1.0
    collapsed = _collapse_hallucinations(raw)
    collapsed = _strip_initial_prompt_echo(collapsed, initial_prompt("hil"))
    if not collapsed or _is_junk_transcript(collapsed):
        return -1.0
    garble = _garble_penalty(collapsed)
    if garble <= 0.2:
        # Treat severe garble as non-viable so a cleaner FW candidate can win.
        return -1.0
    raw_tokens = _norm_tokens(raw)
    tokens = _norm_tokens(collapsed)
    words = len(tokens)
    if words == 0:
        return -1.0
    cov = _segment_time_coverage(segments, duration)
    uniq_ratio = len(set(tokens)) / float(words)
    # How much of the raw text survived collapse — low = heavy hallucination.
    keep_ratio = len(tokens) / float(max(1, len(raw_tokens)))
    if keep_ratio < 0.45:
        return -1.0
    visayan = _visayan_marker_ratio(collapsed)
    visayan_boost = 1.0 + min(0.45, visayan * 2.5)
    # Quiet-mic PH-medium often invents Cebuano filler loops ("adunay daku…").
    low = collapsed.lower()
    filler_hits = low.count("adunay") + low.count("daku nga") + low.count("......")
    filler_penalty = 1.0
    if filler_hits >= 3:
        filler_penalty = max(0.35, 1.0 - 0.15 * filler_hits)
    return (
        float(words)
        * max(0.15, cov)
        * (0.45 + 0.55 * uniq_ratio)
        * (0.35 + 0.65 * keep_ratio)
        * visayan_boost
        * filler_penalty
        * garble
    )


def _sanitize_segments(segments: list[Segment]) -> list[Segment]:
    """Collapse hallucination loops inside segment text before persistence."""
    prompt = initial_prompt("hil")
    out: list[Segment] = []
    for s in segments:
        text = _collapse_hallucinations((s.text or "").strip())
        text = _strip_initial_prompt_echo(text, prompt)
        if not text or _is_junk_transcript(text):
            continue
        # Drop single-glyph / punctuation-only remnants after collapse.
        if len(re.findall(r"[A-Za-zÀ-ÿ0-9]", text)) < 2:
            continue
        # Drop ultra-long segments that are mostly one repeated short phrase.
        duration = max(0.0, float(s.end) - float(s.start))
        words = text.split()
        if duration >= 20.0 and len(words) <= 4:
            continue
        # Severe garble (glued punctuation / fragment spam) should not persist.
        if _garble_penalty(text) <= 0.2:
            continue
        out.append(
            Segment(
                text=text,
                start=s.start,
                end=s.end,
                avg_logprob=getattr(s, "avg_logprob", None),
                no_speech_prob=getattr(s, "no_speech_prob", None),
                low_confidence=bool(getattr(s, "low_confidence", False)),
            )
        )
    return out


def _transcribe_final_hf(
    audio_source,
    language: str | None,
    model_id: str | None = None,
) -> tuple[list[Segment], LanguageDetection | None]:
    """Final pass with a Hugging Face fine-tuned Whisper checkpoint.

    For auto/Hiligaynon meetings the primary decode is Whisper auto-detect
    (never forced Tagalog). Tagalog meetings may retry with forced ``tl``.
    """
    model_id = (model_id or hiligaynon_model_id()).strip()
    cache = get_model_cache()
    pipe = cache.get_hf_pipeline(model_id)
    samples = _audio_to_float32(audio_source)
    if samples.size == 0:
        return [], None

    duration = float(samples.size) / float(settings.audio_sample_rate or 16000)
    primary = _final_decode_language(language)
    lang_attempts: list[str | None] = [primary]
    # Tagalog only: if primary was auto, also try forced native ``tl``.
    # Never append ``tl`` for Hiligaynon.
    forced = _forced_language(language)
    if forced is not None and forced not in lang_attempts:
        lang_attempts.append(forced)

    # Prompt token lists often get echoed (and can truncate PH-medium output).
    # Skip HF prompt_ids for Hiligaynon / PH dialect checkpoints.
    use_prompt = not (
        is_hiligaynon_language(effective_asr_language(language))
        or "whisper-medium-ph" in (model_id or "").lower()
        or "hiligaynon" in (model_id or "").lower()
    )
    prompt_ids = (
        _hf_prompt_ids(pipe, initial_prompt(language, extra_terms=_EXTRA_TERMS_CTX)) if use_prompt else None
    )
    best: list[Segment] = []
    best_score = -1.0
    best_detection: LanguageDetection | None = None
    t0 = time.perf_counter()

    for lang in lang_attempts:
        logger.info(
            "Final Whisper ASR with HF model '%s' (language=%s, task=%s, prompt=%s)",
            model_id,
            lang,
            _WHISPER_TASK,
            bool(prompt_ids),
        )
        gen_kwargs: dict = {"task": _WHISPER_TASK}
        if lang:
            # Whisper-native only (``tl`` / ``en``); never forward ``fil``.
            native = whisper_language_arg(lang)
            if native:
                gen_kwargs["language"] = native
        # Avoid conflict with checkpoint-baked forced_decoder_ids (transformers
        # warning: task/language kwargs otherwise get ignored).
        gen_kwargs["forced_decoder_ids"] = None
        if prompt_ids is not None:
            gen_kwargs["prompt_ids"] = prompt_ids
        try:
            with cache.hf_infer_lock(model_id):
                result = pipe(
                    {
                        "array": samples,
                        "sampling_rate": int(settings.audio_sample_rate),
                    },
                    generate_kwargs=gen_kwargs,
                    return_timestamps=True,
                )
        except TypeError:
            # Older pipeline builds may reject prompt_ids.
            gen_kwargs.pop("prompt_ids", None)
            with cache.hf_infer_lock(model_id):
                result = pipe(
                    {
                        "array": samples,
                        "sampling_rate": int(settings.audio_sample_rate),
                    },
                    generate_kwargs=gen_kwargs,
                    return_timestamps=True,
                )

        segs = _parse_hf_asr_result(result, samples)
        joined = " ".join(s.text for s in segs)
        if not segs or _is_junk_transcript(joined):
            continue
        score = _score_segments(segs, duration)
        if score > best_score:
            best = segs
            best_score = score
            if lang is None:
                best_detection = LanguageDetection(
                    language=None,
                    confidence=None,
                    detected_by="whisper",
                )
            else:
                best_detection = _language_detection_forced(lang)
        # Good enough coverage — skip forced-tl retry.
        if (
            lang is None
            and _segment_time_coverage(segs, duration)
            >= float(settings.whisper_final_min_coverage)
            and sum(len(s.text.split()) for s in segs) >= 8
        ):
            break

    logger.info(
        "asr.final_hf duration_ms=%d samples=%d model=%s best_score=%.1f "
        "detected=%s by=%s",
        int((time.perf_counter() - t0) * 1000),
        int(samples.size),
        model_id,
        best_score if best_score >= 0 else 0.0,
        best_detection.language if best_detection else None,
        best_detection.detected_by if best_detection else None,
    )
    return best, best_detection


def _run_faster_whisper_final(
    model,
    model_id: str,
    audio_in,
    *,
    language: str | None,
    vad_filter: bool,
    meeting_language: str | None = None,
) -> tuple[list[Segment], object]:
    # Sanitize: never pass ``fil``/``hil`` into faster-whisper.
    lang_arg = whisper_language_arg(language) if language else None
    with get_model_cache().fw_infer_lock(model_id):
        segments, info = model.transcribe(
            audio_in,
            language=lang_arg,
            task=_WHISPER_TASK,
            beam_size=5,
            best_of=5,
            temperature=[0.0, 0.2],
            vad_filter=vad_filter,
            vad_parameters=_FINAL_VAD_PARAMS if vad_filter else None,
            # False avoids Whisper latching onto a repeated phrase on quiet PH
            # audio ("Thank you…" / "Ay, wala…" loops). Coverage retries still
            # recover dropped clauses.
            condition_on_previous_text=False,
            without_timestamps=False,
            initial_prompt=initial_prompt(
                meeting_language, extra_terms=_EXTRA_TERMS_CTX
            ),
            # Quiet laptop/board mics need a lower gate or speech is dropped.
            no_speech_threshold=0.25,
            compression_ratio_threshold=2.6,
            log_prob_threshold=-1.2,
        )
    return _segments_to_list(segments), info


def _transcribe_final_faster_whisper(
    audio_source, language: str | None
) -> tuple[list[Segment], LanguageDetection | None]:
    """Final pass via faster-whisper with coverage-aware retries.

    Root cause of missing spoken words on board meetings:
    forcing ``language=tl`` + aggressive VAD skipped long spans of real English
    speech (verified on production WAVs with continuous energy but empty
    transcript gaps of 20–30s). We now default to auto language and no VAD,
    and retry if timeline coverage is still poor.

    Returns ``(segments, language_detection)``.
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

    mode = _final_language_mode(language)
    primary_lang = _final_decode_language(language)
    use_vad = bool(settings.whisper_final_vad_filter)
    min_cov = min(0.95, max(0.2, float(settings.whisper_final_min_coverage)))

    attempts: list[tuple[str, str | None, bool]] = [
        ("primary", primary_lang, use_vad),
    ]
    # Coverage retries — order matters.
    if primary_lang is not None:
        # Tagalog prefer_forced: after forced tl, try auto for code-switched EN.
        attempts.append(("auto_retry", None, use_vad))
    if use_vad:
        attempts.append(("no_vad", primary_lang, False))
        attempts.append(("auto_no_vad", None, False))
    else:
        # Tagalog-only forced retry when primary was auto. Never force tl for hil.
        forced = _forced_language(language)
        if primary_lang is None and forced is not None:
            attempts.append(("forced_lang_retry", forced, False))
        elif primary_lang is not None:
            attempts.append(("auto_retry2", None, False))
    # Tagalog: if primary was auto somehow, still ensure a forced-tl attempt.
    if is_tagalog_language(language) and mode in {"auto", "detect", "none"}:
        attempts.insert(0, ("tagalog_forced_tl", "tl", use_vad))

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
    best_detection: LanguageDetection | None = None
    best_score = (-1.0, -1, -1.0, 0.0)  # weighted words, words, cov, -gap
    t0 = time.perf_counter()
    for label, lang, vad in unique_attempts:
        try:
            segs, info = _run_faster_whisper_final(
                model,
                model_id,
                audio_in,
                language=lang,
                vad_filter=vad,
                meeting_language=language,
            )
        except Exception as exc:
            logger.warning("Final ASR attempt %s failed: %s", label, exc)
            continue
        detection = _language_detection_from_info(info, decode_language=lang)
        if detection.language is None and lang is not None:
            detection.language = _normalize_detected_language(lang)
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
            "confidence=%s by=%s coverage=%.2f gap=%.1fs words=%d "
            "density=%.2f sparse_pen=%.1f",
            label,
            lang,
            vad,
            detection.language,
            detection.confidence,
            detection.detected_by,
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
            best_detection = detection
        # Early exit only when dense + well covered.
        if (
            cov >= max(min_cov, 0.75)
            and gap <= 10.0
            and sparse_penalty < 8.0
            and words >= max(12, int(duration * 0.8))
        ):
            break

    logger.info(
        "asr.final_fw duration_ms=%d model=%s best_score_words_x_cov=%.1f "
        "best_words=%d detected=%s confidence=%s by=%s",
        int((time.perf_counter() - t0) * 1000),
        model_id,
        best_score[0] if best_score[0] >= 0 else 0.0,
        best_score[1] if best_score[1] >= 0 else 0,
        best_detection.language if best_detection else None,
        best_detection.confidence if best_detection else None,
        best_detection.detected_by if best_detection else None,
    )
    return best, best_detection


def _looks_like_hf_repo(model_id: str) -> bool:
    mid = (model_id or "").strip()
    return "/" in mid or mid.startswith(".") or mid.startswith("/")


def _transcribe_final_once(
    audio_source, language: str | None
) -> tuple[list[Segment], LanguageDetection | None]:
    """Single-shot final ASR.

    ``WHISPER_FINAL_BACKEND=auto`` (default) tries Philippine HF candidates,
    scores each non-junk result, compares against faster-whisper, and keeps
    the best transcript — never first-wins on a weak model.
    """
    backend = resolve_final_backend(language)
    configured = (settings.whisper_final_backend or "auto").strip().lower()
    strict_hf = configured in {"huggingface", "hf", "transformers"}
    samples = _audio_to_float32(audio_source)
    duration = (
        float(samples.size) / float(settings.audio_sample_rate or 16000)
        if samples.size
        else 0.0
    )

    best_segs: list[Segment] = []
    best_detection: LanguageDetection | None = None
    best_score = -1.0
    best_mid = ""

    if backend == "huggingface":
        candidates = [
            mid
            for mid in philippine_hf_candidates(language)
            if _looks_like_hf_repo(mid)
        ]
        for model_id in candidates:
            try:
                segs, detection = _transcribe_final_hf(
                    audio_source, language, model_id=model_id
                )
                segs = _sanitize_segments(segs)
                joined = " ".join(s.text for s in segs)
                if not segs or _is_junk_transcript(joined):
                    logger.warning(
                        "HF final ASR '%s' looked like hallucination; "
                        "trying next candidate.",
                        model_id,
                    )
                    continue
                score = _candidate_quality_score(segs, duration)
                logger.info(
                    "HF candidate '%s' score=%.1f words=%d detected=%s by=%s",
                    model_id,
                    score,
                    len(joined.split()),
                    detection.language if detection else None,
                    detection.detected_by if detection else None,
                )
                if score > best_score:
                    best_segs = segs
                    best_detection = detection
                    best_score = score
                    best_mid = model_id
            except TranscriptionUnavailable as exc:
                if strict_hf:
                    raise
                logger.warning(
                    "HF Whisper unavailable (%s); falling back to faster-whisper.",
                    exc,
                )
                break
            except Exception as exc:
                logger.exception(
                    "PH Whisper '%s' failed (%s); trying next / FW fallback.",
                    model_id,
                    exc,
                )

    # Compare faster-whisper unless HF-only mode already has a winner.
    run_fw = (not strict_hf) or best_score < 0
    fw_score = -1.0
    fw_segs: list[Segment] = []
    fw_detection: LanguageDetection | None = None
    fw_mid = ""
    if run_fw:
        try:
            fw_segs, fw_detection = _transcribe_final_faster_whisper(
                audio_source, language
            )
            fw_segs = _sanitize_segments(fw_segs)
            fw_joined = " ".join(s.text for s in fw_segs)
            if fw_segs and not _is_junk_transcript(fw_joined):
                fw_score = _candidate_quality_score(fw_segs, duration)
                fw_mid = f"faster-whisper:{final_faster_model_id(language)}"
                logger.info(
                    "FW candidate score=%.1f words=%d detected=%s by=%s",
                    fw_score,
                    len(fw_joined.split()),
                    fw_detection.language if fw_detection else None,
                    fw_detection.detected_by if fw_detection else None,
                )
                if fw_score > best_score:
                    best_segs = fw_segs
                    best_detection = fw_detection
                    best_score = fw_score
                    best_mid = fw_mid
        except Exception as exc:
            logger.exception("faster-whisper final fallback failed: %s", exc)

    # HF PH checkpoints often emit high-coverage word salad on English-heavy
    # board audio and beat FW on raw word count. Prefer FW unless HF clearly
    # wins — especially when FW detected English or HF looks garbled.
    hf_selected = bool(best_mid) and not str(best_mid).startswith("faster-whisper:")
    if hf_selected and fw_score >= 0 and fw_segs:
        margin = 1.25
        fw_lang = (
            (fw_detection.language or "").strip().lower() if fw_detection else ""
        )
        fw_conf = float(fw_detection.confidence or 0.0) if fw_detection else 0.0
        hf_joined = " ".join(s.text for s in best_segs)
        hf_visayan = _visayan_marker_ratio(hf_joined)
        hf_garble = _garble_penalty(hf_joined)
        prefer_fw = best_score < fw_score * margin
        # Garbled HF output must beat FW by a wide margin.
        if hf_garble < 0.7:
            prefer_fw = best_score < fw_score * 2.5 or hf_garble < 0.45
        # English-majority decode: PH-medium is a poor fit unless it retained
        # real Visayan content.
        if fw_lang in {"en", "english"} and fw_conf >= 0.45 and hf_visayan < 0.04:
            prefer_fw = best_score < fw_score * 2.0 or prefer_fw
        # FW Tagalog/English with solid lexical mass beats PH word salad.
        if fw_lang in {"en", "tl", "tagalog", "fil", "filipino"} and fw_score >= 8.0:
            if hf_garble < 0.75 and best_score < fw_score * 1.8:
                prefer_fw = True
        if prefer_fw:
            logger.info(
                "Final ASR preferring FW '%s' (fw=%.1f) over HF '%s' (hf=%.1f) "
                "margin=%.2f fw_lang=%s hf_visayan=%.3f hf_garble=%.2f",
                fw_mid,
                fw_score,
                best_mid,
                best_score,
                margin,
                fw_lang or None,
                hf_visayan,
                hf_garble,
            )
            best_segs = fw_segs
            best_detection = fw_detection
            best_score = fw_score
            best_mid = fw_mid

    if best_score >= 0 and best_segs:
        logger.info(
            "Final ASR selected '%s' (score=%.1f, meeting_language=%s)",
            best_mid,
            best_score,
            language,
        )
        return best_segs, best_detection

    if strict_hf and best_score < 0:
        return [], None
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


def transcribe_final(
    audio_source,
    language: str | None,
    *,
    extra_terms: list[str] | None = None,
) -> tuple[list[Segment], LanguageDetection | None]:
    """Full-accuracy transcription for Stop Recording / re-transcribe / upload.

    Prefer Philippine HF Whisper candidates (Tagalog + PH medium) when backend
    is auto. Fall back to faster-whisper with **auto language detection** and
    coverage-aware ``tl`` retries.

    Long board-meeting audio (hours) is processed in overlapping chunks so the
    final pass stays within model/memory limits.

    Returns ``(segments, language_detection)``.
    """
    from . import pipeline_metrics, vad as vad_svc

    samples = _prepare_asr_audio(audio_source)
    if samples.size == 0:
        return [], None

    # Soft VAD: if the *entire* recording looks silent, skip Whisper (no
    # hallucinated transcript). Partial silence is left to Whisper's own VAD
    # so we do not crop real speech mid-meeting.
    vad_result = vad_svc.detect_speech(samples, live=False)
    if not vad_result.has_speech:
        logger.info(
            "asr.final vad_skip_all backend=%s %s samples=%d",
            vad_result.backend,
            vad_result.reason,
            int(samples.size),
        )
        return [], None

    # Stash vocab for nested helpers via thread-local-ish attribute on settings
    # is awkward — pass through _transcribe_final_once via closure by setting
    # a module-level context var.
    global _EXTRA_TERMS_CTX
    prev_terms = _EXTRA_TERMS_CTX
    _EXTRA_TERMS_CTX = list(extra_terms or [])
    try:
        with pipeline_metrics.track("asr.final"):
            sr = int(settings.audio_sample_rate)
            duration = float(samples.size) / float(sr)
            chunk_s = max(60.0, float(settings.whisper_final_chunk_seconds))
            overlap_s = max(
                0.0, min(float(settings.whisper_final_chunk_overlap_seconds), chunk_s / 2)
            )

            # Short recordings: single pass.
            if duration <= chunk_s * 1.25:
                return _transcribe_final_once(samples, language)
            # Long recordings: chunked (existing logic below continues...)
            return _transcribe_final_chunked(
                samples, language, duration=duration, chunk_s=chunk_s, overlap_s=overlap_s
            )
    finally:
        _EXTRA_TERMS_CTX = prev_terms


_EXTRA_TERMS_CTX: list[str] = []


def _transcribe_final_chunked(
    samples,
    language: str | None,
    *,
    duration: float,
    chunk_s: float,
    overlap_s: float,
) -> tuple[list[Segment], LanguageDetection | None]:
    """Overlap-chunked final ASR for long board meetings."""
    sr = int(settings.audio_sample_rate)
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
    detection: LanguageDetection | None = None
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
            segs, chunk_detection = _transcribe_final_once(piece, language)
        except Exception:
            logger.exception("Final ASR chunk %d failed; continuing", chunk_idx)
            segs, chunk_detection = [], None
        if chunk_detection and detection is None:
            detection = chunk_detection
        elif (
            chunk_detection
            and detection
            and detection.confidence is None
            and chunk_detection.confidence is not None
        ):
            detection = chunk_detection
        all_segments = _merge_chunk_segments(
            all_segments, segs, time_offset=time_offset
        )
        if end >= samples.size:
            break
        offset += hop_samples

    return all_segments, detection
