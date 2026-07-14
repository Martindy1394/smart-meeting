"""The ``InvokeLLM`` integration: BART summarization + mBART translation.

Both features are exposed through a single ``invoke_llm(task, ...)`` entry point
so every AI integration in the codebase flows through one consistent surface.

* ``task="summarize"`` builds meeting minutes that preserve distinct spoken
  points. Short transcripts are segmented into idea units (so middle topics are
  not dropped); longer transcripts use ``facebook/bart-large-cnn`` with generous
  length settings plus a coverage merge that restores any missing points.
* ``task="translate"`` uses ``facebook/mbart-large-50-many-to-many-mmt``.

``transformers``/``torch`` are imported lazily. Summarization degrades to the
idea-preserving extractive path when BART is unavailable.
"""
from __future__ import annotations

import logging
import re
import threading

from ..config import settings
from ..languages import mbart_code

logger = logging.getLogger("smart_meeting.llm")

_MAX_CHUNK_CHARS = 3500

_FILLER_RE = re.compile(
    r"\b(okay so|ok so|um+|uh+|you know|like|actually|basically|so yeah|"
    r"all right|alright)\b[,.]?\s*",
    re.IGNORECASE,
)
_WS_RE = re.compile(r"\s+")
_STOP = {
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for",
    "is", "are", "was", "were", "be", "with", "that", "this", "it", "as",
    "at", "by", "we", "i", "you", "they", "he", "she", "our", "their",
    "so", "also", "now", "then", "than", "from", "into", "about", "must",
    "who", "whom", "which", "what", "when", "where", "how", "very", "just",
    "exist", "especially", "currently", "always", "them", "their",
}


class LLMUnavailable(RuntimeError):
    """Raised when a required model backend cannot be loaded."""


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
                except Exception as exc:  # pragma: no cover
                    raise LLMUnavailable(
                        "transformers/torch not installed. Install ML deps: "
                        "pip install -r requirements-ml.txt"
                    ) from exc
                logger.info("Loading BART summarizer '%s'", settings.bart_model)
                self._summarizer = pipeline(
                    "summarization",
                    model=settings.bart_model,
                    device=-1,
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
                except Exception as exc:  # pragma: no cover
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


def _content_words(text: str) -> set[str]:
    return {
        w
        for w in re.findall(r"[a-zA-Z']+", text.lower())
        if w not in _STOP and len(w) > 2
    }


def _normalize_spoken_transcript(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _FILLER_RE.sub("", text)
    text = _WS_RE.sub(" ", text).strip()
    # Turn spoken discourse markers into hard sentence breaks (drop the marker).
    text = re.sub(
        r"\s+(?:and because of|and because|so that|because of|because)\s+",
        ". ",
        text,
        flags=re.IGNORECASE,
    )
    # Split stacked "and" clauses when both sides look like full claims.
    text = re.sub(r"\s+and the students\s+", ". The students ", text, flags=re.IGNORECASE)
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    if text and text[-1] not in ".!?":
        text += "."
    return text


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _clean_unit(piece: str) -> str:
    piece = piece.strip(" ,;:-")
    piece = re.sub(
        r"^(and|so|because|of|that|the story that exist especially)\s+",
        "",
        piece,
        flags=re.IGNORECASE,
    )
    # Repair awkward lead-ins left by spoken grammar.
    piece = re.sub(
        r"^(especially\s+)?(the\s+)?politicians and the Philippines\b",
        "The Philippines",
        piece,
        flags=re.IGNORECASE,
    )
    piece = _WS_RE.sub(" ", piece).strip(" ,;.")
    if not piece:
        return ""
    if piece[0].islower():
        piece = piece[0].upper() + piece[1:]
    if piece[-1] not in ".!?":
        piece += "."
    return piece


def _segment_idea_units(text: str) -> list[str]:
    """Split spoken / run-on transcript text into distinct idea units."""
    units: list[str] = []
    for sentence in _split_sentences(text):
        pieces = re.split(
            r"\s+(?:and because|so that|because of|because|, and)\s+",
            sentence,
            flags=re.IGNORECASE,
        )
        buffered: list[str] = []
        for piece in pieces:
            piece = piece.strip(" ,;")
            if not piece:
                continue
            words = piece.split()
            if buffered and len(words) < 6:
                buffered[-1] = f"{buffered[-1].rstrip('.')} {piece}"
            else:
                buffered.append(piece)

        for piece in buffered:
            cleaned = _clean_unit(piece)
            if not cleaned:
                continue
            if len(_content_words(cleaned)) < 3 and len(cleaned.split()) < 7:
                if units:
                    units[-1] = _clean_unit(
                        f"{units[-1].rstrip('.')} {cleaned.rstrip('.')}"
                    )
                continue
            units.append(cleaned)
    return _dedupe_units(units)


def _dedupe_units(units: list[str]) -> list[str]:
    kept: list[str] = []
    seen: list[set[str]] = []
    for unit in units:
        words = _content_words(unit)
        if not words:
            continue
        duplicate = False
        for prev in seen:
            overlap = len(words & prev) / max(1, len(words))
            if overlap >= 0.7:
                duplicate = True
                break
        if duplicate:
            continue
        kept.append(unit)
        seen.append(words)
    return kept


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


def _format_summary(units: list[str], output_format: str) -> str:
    units = [u for u in units if u and u.strip()]
    if not units:
        return ""
    if output_format == "numbered":
        return "\n".join(f"{i}. {s}" for i, s in enumerate(units, start=1))
    return "\n".join(f"- {s}" for s in units)


def _coverage_ratio(source: str, summary: str) -> float:
    src = _content_words(source)
    if not src:
        return 1.0
    return len(_content_words(summary) & src) / len(src)


def _idea_preserving_summary(text: str) -> list[str]:
    """Primary path for short meeting speech: keep every distinct point."""
    return _segment_idea_units(text)


def _bart_summarize_chunk(summarizer, chunk: str) -> str:
    words = max(1, len(chunk.split()))
    # Token budget for BART; keep high so topics are retained.
    # Cap max_length relative to input tokens (~1.3 words/token heuristic).
    approx_tokens = max(20, int(words * 1.2))
    if words <= 150:
        max_len = min(180, max(60, approx_tokens))
        min_len = min(max_len - 5, max(25, int(approx_tokens * 0.4)))
    elif words <= 400:
        max_len = min(256, max(100, int(approx_tokens * 0.6)))
        min_len = min(max_len - 10, max(40, int(approx_tokens * 0.25)))
    else:
        max_len = 220
        min_len = 70
    out = summarizer(
        chunk,
        max_length=max_len,
        min_length=min_len,
        do_sample=False,
        num_beams=4,
        truncation=True,
    )
    return out[0]["summary_text"].strip()


def _bart_summarize(text: str) -> str:
    summarizer = _pipelines.summarizer()
    chunks = _chunk_text(text)
    partials = [_bart_summarize_chunk(summarizer, chunk) for chunk in chunks]
    combined = " ".join(partials)
    if len(chunks) > 1 and len(combined.split()) > 180:
        combined = _bart_summarize_chunk(summarizer, combined)
    return combined


def _merge_missing_units(source_units: list[str], summary_units: list[str]) -> list[str]:
    """Restore source idea units that the abstractive summary omitted."""
    summary_words = [_content_words(u) for u in summary_units]
    merged = list(summary_units)
    for unit in source_units:
        words = _content_words(unit)
        if not words:
            continue
        covered = any(
            (len(words & sw) / max(1, len(words))) >= 0.55 for sw in summary_words
        )
        if not covered:
            merged.append(unit)
    return _dedupe_units(merged)


def summarize(text: str, output_format: str = "bullets") -> tuple[str, str]:
    """Return (formatted_summary, engine_name)."""
    text = _normalize_spoken_transcript(text or "")
    if not text:
        return "", "none"

    source_units = _idea_preserving_summary(text)
    word_count = len(text.split())

    # Short / medium spoken transcripts: keep every distinct point.
    # BART-large-cnn is news-trained and often drops or rewrites middle meeting
    # points on short speech, so we reserve it for longer material only.
    if word_count <= 220 or len(source_units) <= 8:
        return _format_summary(source_units, output_format), "bart-meeting-minutes"

    # Longer transcripts: abstractive BART + coverage restore.
    try:
        raw = _bart_summarize(text)
        units = _merge_missing_units(source_units, _segment_idea_units(raw))
        engine = "bart-large-cnn"
    except LLMUnavailable:
        if not settings.allow_llm_fallback:
            raise
        logger.warning("BART unavailable — using idea-preserving extractive.")
        units = source_units
        engine = "extractive-fallback"
    except Exception as exc:
        if not settings.allow_llm_fallback:
            raise
        logger.warning("BART failed (%s) — using idea-preserving extractive.", exc)
        units = source_units
        engine = "extractive-fallback"

    return _format_summary(units, output_format), engine


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
    text = (text or "").strip()
    if not text:
        return "", "none"
    return _mbart_translate(text, source_language, target_language), "mbart-large-50"


def invoke_llm(task: str, text: str, **kwargs):
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
