"""The ``InvokeLLM`` integration: BART summarization + mBART translation.

Both features are exposed through a single ``invoke_llm(task, ...)`` entry point
so every AI integration in the codebase flows through one consistent surface.

* ``task="summarize"`` uses ``facebook/bart-large-cnn`` and post-processes the
  abstractive summary into the requested output format — bullet points or
  numbered sentences within paragraphs — deterministically, guaranteeing the
  format is respected exactly.
* ``task="translate"`` uses ``facebook/mbart-large-50-many-to-many-mmt`` with the
  correct source/target language tokens, returning only the translated text.

``transformers``/``torch`` are imported lazily.  Summarization degrades to a
lightweight extractive fallback when the models are unavailable (controlled by
``ALLOW_LLM_FALLBACK``); translation requires the real model.
"""
from __future__ import annotations

import logging
import re
import threading

from ..config import settings
from ..languages import language_name, mbart_code

logger = logging.getLogger("smart_meeting.llm")

# Rough max input the summarizer accepts in one shot; longer transcripts are
# chunked and their partial summaries recombined.
_MAX_CHUNK_CHARS = 3500


class LLMUnavailable(RuntimeError):
    """Raised when a required model backend cannot be loaded."""


# --------------------------------------------------------------------------- #
# Model caches (lazy, thread-safe)
# --------------------------------------------------------------------------- #
class _Pipelines:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._summarizer = None
        self._mbart_model = None
        self._mbart_tokenizer = None

    def summarizer(self):
        with self._lock:
            if self._summarizer is None:
                try:
                    from transformers import pipeline  # type: ignore
                except Exception as exc:  # pragma: no cover - optional dep
                    raise LLMUnavailable(
                        "transformers/torch not installed. Install ML deps: "
                        "pip install -r requirements-ml.txt"
                    ) from exc
                logger.info("Loading BART summarizer '%s'", settings.bart_model)
                self._summarizer = pipeline(
                    "summarization", model=settings.bart_model
                )
            return self._summarizer

    def mbart(self):
        with self._lock:
            if self._mbart_model is None:
                try:
                    from transformers import (  # type: ignore
                        MBart50TokenizerFast,
                        MBartForConditionalGeneration,
                    )
                except Exception as exc:  # pragma: no cover - optional dep
                    raise LLMUnavailable(
                        "transformers/torch not installed. Install ML deps: "
                        "pip install -r requirements-ml.txt"
                    ) from exc
                logger.info("Loading mBART translator '%s'", settings.mbart_model)
                self._mbart_tokenizer = MBart50TokenizerFast.from_pretrained(
                    settings.mbart_model
                )
                self._mbart_model = MBartForConditionalGeneration.from_pretrained(
                    settings.mbart_model
                )
            return self._mbart_model, self._mbart_tokenizer


_pipelines = _Pipelines()


# --------------------------------------------------------------------------- #
# Text utilities
# --------------------------------------------------------------------------- #
def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _chunk_text(text: str, size: int = _MAX_CHUNK_CHARS) -> list[str]:
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    chunks: list[str] = []
    current = ""
    for sentence in _split_sentences(text):
        if len(current) + len(sentence) + 1 > size and current:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current.strip())
    return chunks


def _format_summary(summary_text: str, output_format: str) -> str:
    sentences = _split_sentences(summary_text)
    if not sentences:
        return ""
    if output_format == "numbered":
        return " ".join(f"{i}. {s}" for i, s in enumerate(sentences, start=1))
    # Default: bullet points, one per line.
    return "\n".join(f"- {s}" for s in sentences)


# --------------------------------------------------------------------------- #
# Fallback (no ML deps) — extractive summarization
# --------------------------------------------------------------------------- #
def _extractive_summary(text: str, max_sentences: int = 6) -> str:
    sentences = _split_sentences(text)
    if len(sentences) <= max_sentences:
        return " ".join(sentences)
    # Simple frequency-based scoring.
    words = re.findall(r"[a-zA-Z']+", text.lower())
    stop = {
        "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for",
        "is", "are", "was", "were", "be", "with", "that", "this", "it", "as",
        "at", "by", "we", "i", "you", "they", "he", "she", "our", "their",
    }
    freq: dict[str, int] = {}
    for w in words:
        if w not in stop and len(w) > 2:
            freq[w] = freq.get(w, 0) + 1
    scored = []
    for idx, sentence in enumerate(sentences):
        s_words = re.findall(r"[a-zA-Z']+", sentence.lower())
        score = sum(freq.get(w, 0) for w in s_words)
        score = score / (len(s_words) + 1)
        scored.append((score, idx, sentence))
    top = sorted(scored, key=lambda x: x[0], reverse=True)[:max_sentences]
    top_sorted = [s for _, _, s in sorted(top, key=lambda x: x[1])]
    return " ".join(top_sorted)


# --------------------------------------------------------------------------- #
# Summarization
# --------------------------------------------------------------------------- #
def _bart_summarize(text: str) -> str:
    summarizer = _pipelines.summarizer()
    chunks = _chunk_text(text)
    partials: list[str] = []
    for chunk in chunks:
        words = len(chunk.split())
        max_len = max(40, min(180, words // 2))
        min_len = max(15, max_len // 3)
        out = summarizer(
            chunk, max_length=max_len, min_length=min_len, do_sample=False
        )
        partials.append(out[0]["summary_text"].strip())
    combined = " ".join(partials)
    # Second condensation pass when the transcript was long.
    if len(chunks) > 1 and len(combined) > 600:
        out = summarizer(combined, max_length=180, min_length=40, do_sample=False)
        combined = out[0]["summary_text"].strip()
    return combined


def summarize(text: str, output_format: str = "bullets") -> tuple[str, str]:
    """Return (formatted_summary, engine_name)."""
    text = (text or "").strip()
    if not text:
        return "", "none"
    try:
        raw = _bart_summarize(text)
        engine = "bart-large-cnn"
    except LLMUnavailable:
        if not settings.allow_llm_fallback:
            raise
        logger.warning("BART unavailable — using extractive fallback summarizer.")
        raw = _extractive_summary(text)
        engine = "extractive-fallback"
    return _format_summary(raw, output_format), engine


# --------------------------------------------------------------------------- #
# Translation
# --------------------------------------------------------------------------- #
def _mbart_translate(text: str, src_code: str, tgt_code: str) -> str:
    model, tokenizer = _pipelines.mbart()
    src = mbart_code(src_code) or "en_XX"
    tgt = mbart_code(tgt_code)
    if not tgt:
        raise LLMUnavailable(f"Unsupported target language: {tgt_code}")

    outputs: list[str] = []
    for chunk in _chunk_text(text):
        tokenizer.src_lang = src
        encoded = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=1024)
        generated = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt],
            max_length=1024,
            num_beams=5,
        )
        outputs.append(
            tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        )
    return " ".join(outputs).strip()


def translate(text: str, target_language: str, source_language: str = "en") -> tuple[str, str]:
    """Return (translated_text, engine_name)."""
    text = (text or "").strip()
    if not text:
        return "", "none"
    translated = _mbart_translate(text, source_language, target_language)
    return translated, "mbart-large-50"


# --------------------------------------------------------------------------- #
# Unified integration entry point
# --------------------------------------------------------------------------- #
def invoke_llm(task: str, text: str, **kwargs):
    """Single integration surface for all AI features.

    tasks:
      - "summarize": kwargs: output_format ("bullets"|"numbered")
      - "translate": kwargs: target_language, source_language
    """
    if task == "summarize":
        return summarize(text, output_format=kwargs.get("output_format", "bullets"))
    if task == "translate":
        return translate(
            text,
            target_language=kwargs["target_language"],
            source_language=kwargs.get("source_language", "en"),
        )
    raise ValueError(f"Unknown InvokeLLM task: {task}")


def summarizer_available() -> bool:
    try:
        import transformers  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False
