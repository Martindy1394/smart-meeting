"""The ``InvokeLLM`` integration: BART summarization + translation.

Both features are exposed through a single ``invoke_llm(task, ...)`` entry point
so every AI integration in the codebase flows through one consistent surface.

* Meeting summarization (``summarize_to_english``) is a two-step pipeline:
  (1) three-way MT — English passthrough / Tagalog→NLLB / Hiligaynon→Google
  Translate, (2) topic-aware BART on that English.
* ``source_kind`` / content kind: ``meeting`` (board framing + Discussion /
  Decisions / Action items) or ``general`` (neutral framing + flat/topic
  bullets, no minutes bucketing). ``english_translation`` aliases ``meeting``.
* ``task="translate"`` uses the same three-way router for →English.
"""
from __future__ import annotations

import logging
import math
import re
import threading
from collections import Counter
from dataclasses import dataclass, field

from ..config import settings
from ..languages import mbart_code

logger = logging.getLogger("smart_meeting.llm")

_MAX_CHUNK_CHARS = 3500
_SPEAKER_TURN_RE = re.compile(
    r"^(?:Speaker\s*\d+|[\w][\w .'-]{0,40})\s*:\s+(.+)$",
    re.IGNORECASE,
)

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
    # Common Filipino function words (keep out of content-word overlap checks).
    "ang", "mga", "ng", "nang", "sa", "ay", "na", "lang", "naman", "po",
    "ba", "pala", "daw", "raw", "yung", "yun", "ito", "iyan", "iyon",
    "ko", "mo", "niya", "nila", "natin", "namin", "kayo", "kami", "tayo",
    "ako", "siya", "sila", "may", "mayroon", "meron", "wala", "para",
    "kung", "kapag", "kasi", "kaya", "pero", "at", "o", "dahil",
}

# Filipino / Tagalog sentence starters often glued onto English in Whisper output.
_FILIPINO_STARTER_RE = re.compile(
    r"(?<=[A-Za-z0-9,\"'”’])\s+(?=("
    r"Bakit|Kasi|Kaya|Pero|Hindi|Huag|Huwag|Ngayon|Tapos|Ganon|Ganun|"
    r"Wala|Meron|Mayroon|Dapat|Sana|Ako|Ikaw|Kami|Tayo|Sila|Mga|"
    r"Ang|Sa|Para|Dahil|Kapag|Kung|Lagi|Talaga|Sige|Oo|"
    r"Pangarap|Mahal|Gusto|Kailangan"
    r")\b)",
    re.IGNORECASE,
)

_FILIPINO_MARKERS = frozenset(
    {
        "ang", "mga", "nang", "ay", "ba", "po", "opo", "naman", "lang",
        "kasi", "kaya", "pero", "bakit", "hindi", "wala", "meron", "mayroon",
        "dapat", "sana", "tayo", "kami", "sila", "ako", "ikaw", "niya",
        "nila", "natin", "yung", "iyan", "iyon", "dito", "doon", "roon",
        "pilipino", "pilipinos", "politiko", "maging", "naging", "naginging",
        "lagi", "talaga", "ganun", "ganon", "para", "dahil", "kapag",
        # Common Tagalog content words that should force translation.
        "pangarap", "ibigin", "habang", "panahon", "makasama", "buhay",
        "kulang", "siyang", "ito", "ngayon", "tapos", "salamat", "mahal",
        "gusto", "kailangan", "pwede", "puwede", "sige", "oo", "huwag",
        "kanila", "kanilang", "atin", "inyo", "ninyo", "kayo",
    }
)
_ENGLISH_MARKERS = frozenset(
    {
        "the", "and", "you", "that", "with", "this", "have", "from", "they",
        "were", "was", "are", "is", "my", "your", "our", "when", "young",
        "everybody", "watching", "reminds", "movie", "song", "sound", "check",
        "things", "talk", "move", "hoping", "loves", "guy",
        "mind", "change", "don't", "dont", "tonight", "again", "hold",
        "breath", "know", "confused", "fall", "over", "because", "will",
        "make", "me", "my", "core", "down",
    }
)


class LLMUnavailable(RuntimeError):
    """Raised when a required model backend cannot be loaded."""


class _Pipelines:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Fast tokenizers are not thread-safe ("Already borrowed" under concurrency).
        self._mbart_infer_lock = threading.Lock()
        self._nllb_infer_lock = threading.Lock()
        self._summarizer_infer_lock = threading.Lock()
        self._summarizer = None
        self._mbart_model = None
        self._mbart_tokenizer = None
        self._nllb_model = None
        self._nllb_tokenizer = None

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
                model_id = (settings.mbart_ph_finetuned_model or "").strip() or (
                    settings.mbart_model
                )
                logger.info("Loading mBART translator '%s'", model_id)
                self._mbart_tokenizer = MBart50TokenizerFast.from_pretrained(model_id)
                self._mbart_model = MBartForConditionalGeneration.from_pretrained(
                    model_id
                )
                # Vocab-extended hil_XX fine-tunes need lang_code_to_id rebound.
                from ..languages import ensure_tokenizer_hil_xx

                if ensure_tokenizer_hil_xx(self._mbart_tokenizer):
                    logger.info("mBART tokenizer has hil_XX (PH fine-tune vocab)")
            return self._mbart_model, self._mbart_tokenizer
    def nllb(self):
        with self._lock:
            if self._nllb_model is None:
                try:
                    from transformers import (  # type: ignore
                        AutoModelForSeq2SeqLM,
                        AutoTokenizer,
                    )
                except Exception as exc:  # pragma: no cover
                    raise LLMUnavailable(
                        "transformers/torch not installed. Install ML deps: "
                        "pip install -r requirements-ml.txt"
                    ) from exc
                model_id = (settings.nllb_model or "").strip() or (
                    "facebook/nllb-200-distilled-600M"
                )
                logger.info("Loading NLLB translator '%s'", model_id)
                self._nllb_tokenizer = AutoTokenizer.from_pretrained(model_id)
                self._nllb_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            return self._nllb_model, self._nllb_tokenizer

    def mbart_infer_lock(self) -> threading.Lock:
        return self._mbart_infer_lock

    def nllb_infer_lock(self) -> threading.Lock:
        return self._nllb_infer_lock

    def summarizer_infer_lock(self) -> threading.Lock:
        return self._summarizer_infer_lock


_pipelines = _Pipelines()


def _content_words(text: str) -> set[str]:
    return {
        w
        for w in re.findall(r"[A-Za-zÀ-ÿ']+", text.lower())
        if w not in _STOP and len(w) > 2
    }


def _language_scores(text: str) -> tuple[float, float]:
    """Return (english_score, filipino_score) from simple marker hits."""
    tokens = re.findall(r"[A-Za-zÀ-ÿ']+", text.lower())
    if not tokens:
        return 0.0, 0.0
    en = sum(1 for t in tokens if t in _ENGLISH_MARKERS)
    fi = sum(1 for t in tokens if t in _FILIPINO_MARKERS)
    n = max(1, len(tokens))
    return en / n, fi / n


# English clause after Filipino (Whisper often glues: "...ibigin ka I know you").
_ENGLISH_AFTER_PH_RE = re.compile(
    r"(?<=[A-Za-zÀ-ÿ,\"'”’])\s+(?=("
    r"I|I'm|I've|I'd|I'll|You|We|They|He|She|The|This|That|These|Those|"
    r"What|When|Where|Why|How|Who|Don't|Doesn't|Didn't|Can't|Won't|"
    r"Tonight|Today|Tomorrow|Because|Over|Hold|Please|Thank|Hello|Good|"
    r"My|Your|Our|His|Her|Their"
    r")\b)",
    re.IGNORECASE,
)


def _insert_context_breaks(text: str) -> str:
    """Insert sentence breaks where Whisper glued different spoken contexts.

    Walk left-to-right so an English→Filipino split becomes a new clause
    boundary before later Filipino function words (ang/mga/lagi) are considered.
    Also split Filipino→English glued turns.
    """
    fil_func = _FILIPINO_MARKERS | {
        "ba", "lang", "na", "ang", "mga", "ng", "sa", "ay", "po", "naman", "pa",
        "haa", "ha",
    }
    pieces: list[str] = []
    cursor = 0
    for match in _FILIPINO_STARTER_RE.finditer(text):
        # Match is the whitespace between a prior token and a Filipino starter.
        ws_start, ws_end = match.start(), match.end()
        pieces.append(text[cursor:ws_start])
        built = "".join(pieces)
        clause = re.split(r"[.!?]\s*", built)[-1]
        prev_tokens = re.findall(r"[A-Za-zÀ-ÿ']+", clause.lower())
        last_tok = prev_tokens[-1] if prev_tokens else ""
        if last_tok in fil_func:
            pieces.append(text[ws_start:ws_end])
        else:
            en, fi = _language_scores(clause)
            # Split when the previous token is English (code-switch boundary),
            # even if the longer clause still scores as Filipino overall.
            if last_tok in _ENGLISH_MARKERS or (
                prev_tokens and en >= fi and en > 0
            ):
                pieces.append(". ")
            else:
                pieces.append(text[ws_start:ws_end])
        cursor = ws_end
    pieces.append(text[cursor:])
    text = "".join(pieces)

    # Second pass: Filipino → English glued clauses.
    pieces = []
    cursor = 0
    for match in _ENGLISH_AFTER_PH_RE.finditer(text):
        ws_start, ws_end = match.start(), match.end()
        pieces.append(text[cursor:ws_start])
        built = "".join(pieces)
        clause = re.split(r"[.!?]\s*", built)[-1]
        en, fi = _language_scores(clause)
        if fi > en and fi >= 0.08:
            pieces.append(". ")
        else:
            pieces.append(text[ws_start:ws_end])
        cursor = ws_end
    pieces.append(text[cursor:])
    text = "".join(pieces)

    text = re.sub(
        r"\s+(?:meanwhile|on another note|separately|next topic|"
        r"moving on|another thing|also note that)\s+",
        ". ",
        text,
        flags=re.IGNORECASE,
    )
    return text


def _normalize_spoken_transcript(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _FILLER_RE.sub("", text)
    text = _WS_RE.sub(" ", text).strip()
    text = _insert_context_breaks(text)
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


def _split_mixed_language_unit(unit: str) -> list[str]:
    """Split one unit when English and Filipino contexts were concatenated."""
    en, fi = _language_scores(unit)
    # Only attempt when both languages are clearly present.
    if en < 0.08 or fi < 0.08:
        return [unit]

    tokens = unit.split()
    if len(tokens) < 8:
        return [unit]

    best_i = None
    best_score = 0.0
    starter_bonus = {
        "bakit", "kasi", "kaya", "pero", "hindi", "ngayon", "tapos",
        "ganun", "ganon", "wala", "meron", "dapat", "sana", "lagi",
    }
    for i in range(3, len(tokens) - 2):
        left = " ".join(tokens[:i])
        right = " ".join(tokens[i:])
        len_, lif = _language_scores(left)
        ren, rif = _language_scores(right)
        # Prefer English→Filipino or Filipino→English contrast.
        score = abs((len_ - lif) - (ren - rif))
        if (len_ > lif and rif > ren) or (lif > len_ and ren > rif):
            score += 0.35
        head = tokens[i].lower().rstrip(".,!?")
        if head in starter_bonus:
            score += 0.55
        if score > best_score:
            best_score = score
            best_i = i

    if best_i is None or best_score < 0.55:
        return [unit]

    left = _clean_unit(" ".join(tokens[:best_i]))
    right = _clean_unit(" ".join(tokens[best_i:]))
    out = [u for u in (left, right) if u]
    # Recurse once in case more than two contexts were glued.
    if len(out) == 2:
        more: list[str] = []
        for part in out:
            split = _split_mixed_language_unit(part)
            # Guard against infinite recursion on tiny leftovers.
            if split == [part] or len(part.split()) < 10:
                more.append(part)
            else:
                more.extend(split)
        return more
    return out or [unit]


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
            en, fi = _language_scores(piece)
            language_shift = False
            if buffered:
                prev_en, prev_fi = _language_scores(buffered[-1])
                language_shift = (prev_en > prev_fi and fi > en + 0.05) or (
                    prev_fi > prev_en and en > fi + 0.05
                )
            if buffered and len(words) < 6 and not language_shift:
                buffered[-1] = f"{buffered[-1].rstrip('.')} {piece}"
            else:
                buffered.append(piece)

        for piece in buffered:
            cleaned = _clean_unit(piece)
            if not cleaned:
                continue
            for part in _split_mixed_language_unit(cleaned):
                if len(_content_words(part)) < 3 and len(part.split()) < 7:
                    if units:
                        u_en, u_fi = _language_scores(units[-1])
                        p_en, p_fi = _language_scores(part)
                        if (u_en > u_fi and p_fi > p_en) or (
                            u_fi > u_en and p_en > p_fi
                        ):
                            units.append(part)
                        else:
                            units[-1] = _clean_unit(
                                f"{units[-1].rstrip('.')} {part.rstrip('.')}"
                            )
                    else:
                        units.append(part)
                    continue
                units.append(part)
    return _dedupe_units(units)


# Dedup: require enough shared content words so short lyric lines that share
# thematic vocabulary (ulan/puso/damdamin) are not collapsed as ASR repeats.
_DEDUPE_MIN_SHARED_WORDS = 3
_DEDUPE_RATIO = 0.7
# Real ASR duplication is adjacent; only compare against this many prior units.
_DEDUPE_LOCAL_WINDOW = 2


def _dedupe_units(units: list[str]) -> list[str]:
    """Drop near-duplicate units from literal ASR repetition.

    Uses a local window + minimum absolute overlap so thematic recurrence in
    song lyrics (distinct lines sharing a few content words) is preserved.
    """
    kept: list[str] = []
    seen: list[set[str]] = []
    for unit in units:
        words = _content_words(unit)
        if not words:
            # Keep non-empty surface text that has no content words (rare);
            # skip pure noise like "Mga m." with no usable tokens.
            if (unit or "").strip() and len((unit or "").split()) >= 3:
                kept.append(unit)
                seen.append(set())
            continue
        duplicate = False
        # Adjacent-only: ASR loops repeat next to each other, not across a verse.
        for prev in seen[-_DEDUPE_LOCAL_WINDOW:]:
            shared = len(words & prev)
            if shared < _DEDUPE_MIN_SHARED_WORDS:
                continue
            overlap = shared / max(1, len(words))
            if overlap >= _DEDUPE_RATIO:
                duplicate = True
                break
        if duplicate:
            continue
        kept.append(unit)
        seen.append(words)
    return kept


def _chunk_text(
    text: str,
    size: int = _MAX_CHUNK_CHARS,
    *,
    overlap_chars: int = 0,
) -> list[str]:
    """Sentence-aware character chunker for mBART (optional overlap)."""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    overlap_chars = max(0, min(int(overlap_chars), size // 2))
    sentences = _split_sentences(text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 > size and current:
            chunks.append(current.strip())
            if overlap_chars > 0:
                # Keep a trailing overlap window so discourse spans chunk edges.
                tail = current[-overlap_chars:]
                # Prefer starting overlap at a word boundary.
                sp = tail.find(" ")
                current = (tail[sp + 1 :] if sp >= 0 else tail).strip()
                current = f"{current} {sentence}".strip() if current else sentence
            else:
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
    # Real bullet points (not ASCII hyphens).
    return "\n".join(f"• {s}" for s in units)


def _strip_list_prefix(text: str) -> str:
    return re.sub(r"^(?:[-*•]|\d+[.)])\s+", "", text.strip())


def _summary_sentences_to_units(summary: str) -> list[str]:
    """Turn a paragraph-style BART summary into bullet-ready sentences."""
    units: list[str] = []
    for sentence in _split_sentences(summary):
        cleaned = _clean_unit(_strip_list_prefix(sentence))
        if cleaned and len(_content_words(cleaned)) >= 2:
            units.append(cleaned)
    if units:
        return _dedupe_units(units)
    # Fallback: idea-unit segmentation if BART returned one long clause.
    return _segment_idea_units(summary)


def _discourse_units(text: str) -> list[str]:
    """Split transcript into speaker turns when labeled, else sentences."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    turns: list[str] = []
    if lines and sum(1 for ln in lines if _SPEAKER_TURN_RE.match(ln)) >= max(
        2, len(lines) // 2
    ):
        for ln in lines:
            m = _SPEAKER_TURN_RE.match(ln)
            turns.append((m.group(1) if m else ln).strip())
        return [t for t in turns if t]

    # Prefer idea units (handles run-on ASR) over raw sentence splits.
    units = _segment_idea_units(text)
    return units or _split_sentences(text)


def _bow_vector(text: str) -> Counter[str]:
    return Counter(_content_words(text))


def _cosine_similarity(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _estimate_tokens(text: str, tokenizer=None) -> int:
    text = (text or "").strip()
    if not text:
        return 0
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    # ~0.75 words/token heuristic for English-ish ASR text.
    return max(1, int(len(text.split()) / 0.75))


def _hard_split_unit(unit: str, max_tokens: int, tokenizer=None) -> list[str]:
    """Split an oversized unit by words so it fits the BART token budget."""
    words = unit.split()
    if not words:
        return []
    if _estimate_tokens(unit, tokenizer) <= max_tokens:
        return [unit]
    parts: list[str] = []
    current: list[str] = []
    for word in words:
        trial = " ".join(current + [word])
        if current and _estimate_tokens(trial, tokenizer) > max_tokens:
            parts.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        parts.append(" ".join(current))
    return parts


def _topic_label(chunk: str, index: int) -> str:
    words = [w for w in re.findall(r"[A-Za-zÀ-ÿ0-9']+", chunk) if w.lower() not in _STOP]
    if not words:
        return f"Topic {index}"
    label = " ".join(words[:6])
    if len(label) > 48:
        label = label[:45].rstrip() + "…"
    return label[0].upper() + label[1:] if label else f"Topic {index}"


def segment_transcript_topics(
    text: str,
    *,
    max_tokens: int | None = None,
    similarity_threshold: float | None = None,
    min_units: int | None = None,
    tokenizer=None,
) -> list[str]:
    """Split a long transcript into topic-sized chunks for BART.

    Uses chunked linear accumulation of discourse units (speaker turns or
    idea sentences). Starts a new topic when consecutive-unit TF cosine
    similarity drops below ``similarity_threshold``, or when the BART token
    budget would be exceeded. Never cuts mid-unit unless a single unit is
    larger than the budget.
    """
    budget = max_tokens if max_tokens is not None else settings.bart_max_input_tokens
    threshold = (
        similarity_threshold
        if similarity_threshold is not None
        else settings.bart_topic_similarity_threshold
    )
    min_u = min_units if min_units is not None else settings.bart_topic_min_units
    budget = max(128, int(budget))

    units: list[str] = []
    for unit in _discourse_units(text):
        units.extend(_hard_split_unit(unit, budget, tokenizer))
    if not units:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_vec: Counter[str] | None = None

    def flush() -> None:
        nonlocal current, current_vec
        if current:
            chunks.append(" ".join(current).strip())
            current = []
            current_vec = None

    for unit in units:
        unit_tokens = _estimate_tokens(unit, tokenizer)
        unit_vec = _bow_vector(unit)
        if not current:
            current = [unit]
            current_vec = unit_vec
            continue

        joined = " ".join(current + [unit])
        joined_tokens = _estimate_tokens(joined, tokenizer)
        sim = _cosine_similarity(current_vec or Counter(), unit_vec)
        topic_break = len(current) >= min_u and sim < threshold
        over_budget = joined_tokens > budget

        if topic_break or over_budget:
            flush()
            current = [unit]
            current_vec = unit_vec
            # Extremely rare: single unit still over budget after hard split.
            if unit_tokens > budget:
                flush()
            continue

        current.append(unit)
        if current_vec is None:
            current_vec = unit_vec
        else:
            current_vec = current_vec + unit_vec

    flush()
    return [c for c in chunks if c]


def _format_topic_summaries(
    topic_sections: list[tuple[str, list[str]]],
    output_format: str,
) -> str:
    """Format per-topic bullet/numbered lists, with headings when multi-topic."""
    if not topic_sections:
        return ""
    show_headers = len(topic_sections) > 1
    blocks: list[str] = []
    counter = 1
    for label, units in topic_sections:
        units = [u for u in units if u and u.strip()]
        if not units:
            continue
        lines: list[str] = []
        if show_headers:
            lines.append(label)
        for unit in units:
            if output_format == "numbered":
                lines.append(f"{counter}. {unit}")
                counter += 1
            else:
                lines.append(f"• {unit}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


_DECISION_RE = re.compile(
    r"\b("
    r"decid(?:e|ed|es|ing)|approv(?:e|ed|es|ing|al)|agree(?:d|s|ment)?|"
    r"resolv(?:e|ed|ution)|motion\s+carried|passed|confirmed|adopted|"
    r"finaliz(?:e|ed)|ratif(?:y|ied)|authorized|endorsed"
    r")\b",
    re.IGNORECASE,
)
_ACTION_RE = re.compile(
    r"\b("
    r"will|shall|need(?:s)?\s+to|must|should|assign(?:ed|ment)?|"
    r"follow[\s-]?up|action\s+items?|responsible|deadline|by\s+next|"
    r"schedule(?:d|s)?|prepare|submit|send|contact|implement|complete|"
    r"deliver|coordinate|review\s+and|take\s+care\s+of"
    r")\b",
    re.IGNORECASE,
)


def _classify_minute_unit(unit: str) -> str:
    """Bucket a summary unit into meeting-minutes sections."""
    text = unit or ""
    if _DECISION_RE.search(text):
        return "Decisions"
    if _ACTION_RE.search(text):
        return "Action items"
    return "Discussion"


def _format_meeting_minutes(
    units: list[str],
    output_format: str,
    *,
    content_kind: str = "meeting",
) -> str:
    """Structure bullets as Discussion / Decisions / Action items.

    Only for ``content_kind="meeting"``. Non-meeting kinds return a flat
    bullet list so ordinary future-tense phrasing is not force-bucketed into
    Action items / Decisions.
    """
    if _resolve_content_kind(content_kind) != "meeting":
        return _format_summary(units, output_format)
    cleaned = [u.strip() for u in units if u and u.strip()]
    if not cleaned:
        return ""
    buckets: dict[str, list[str]] = {
        "Discussion": [],
        "Decisions": [],
        "Action items": [],
    }
    for unit in cleaned:
        buckets[_classify_minute_unit(unit)].append(unit)

    # Single-bucket short notes stay as a flat list (less chrome).
    nonempty = [k for k, v in buckets.items() if v]
    if len(nonempty) <= 1 and len(cleaned) <= 4:
        return _format_summary(cleaned, output_format)

    blocks: list[str] = []
    counter = 1
    for label in ("Discussion", "Decisions", "Action items"):
        items = buckets[label]
        if not items:
            continue
        lines = [label]
        for unit in items:
            if output_format == "numbered":
                lines.append(f"{counter}. {unit}")
                counter += 1
            else:
                lines.append(f"• {unit}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _coverage_ratio(source: str, summary: str) -> float:
    src = _content_words(source)
    if not src:
        return 1.0
    return len(_content_words(summary) & src) / len(src)


# Kept-source / untranslated marker — visible in the English panel so readers
# can tell MT declined rather than treating Tagalog as English gloss.
# Downstream: strip with ``strip_untranslated_spans`` before BART; faithfulness
# surfaces each span under section "Untranslated".
_UNTRANSLATED_OPEN = "[untranslated:"
_UNTRANSLATED_CLOSE = "]"
_UNTRANSLATED_RE = re.compile(
    r"\[untranslated:\s*(.*?)\s*\]",
    re.IGNORECASE | re.DOTALL,
)
# Flag when translated PH token-mass drops below this vs source PH units.
_TRANSLATION_MASS_COVERAGE_MIN = 0.50


def mark_untranslated(unit: str) -> str:
    """Wrap a kept-source clause so it is visible in joined English output."""
    inner = _WS_RE.sub(" ", (unit or "").strip())
    if not inner:
        return ""
    # Neutralize nested brackets so the span stays machine-parseable.
    inner = inner.replace("[", "(").replace("]", ")")
    return f"{_UNTRANSLATED_OPEN} {inner}{_UNTRANSLATED_CLOSE}"


def iter_untranslated_spans(text: str) -> list[str]:
    return [m.group(1).strip() for m in _UNTRANSLATED_RE.finditer(text or "") if m.group(1).strip()]


def strip_untranslated_spans(text: str, *, keep_inner: bool = False) -> str:
    """Remove ``[untranslated: …]`` markers from text.

    ``keep_inner=False`` (default) drops the untranslated source entirely —
    used before BART so Tagalog ASR junk is not summarized as English.
    ``keep_inner=True`` leaves the raw source text without the wrapper.
    """
    if keep_inner:
        out = _UNTRANSLATED_RE.sub(lambda m: m.group(1).strip(), text or "")
    else:
        out = _UNTRANSLATED_RE.sub(" ", text or "")
    return _WS_RE.sub(" ", out).strip()


def _token_mass(text: str) -> int:
    """Count tokens length>2 — language-agnostic mass for coverage checks."""
    return sum(1 for w in re.findall(r"[A-Za-zÀ-ÿ']+", (text or "").lower()) if len(w) > 2)


def _translation_mass_coverage(source_mass: int, translated_mass: int) -> float:
    if source_mass <= 0:
        return 1.0
    return translated_mass / float(source_mass)


def assess_minutes_faithfulness(
    summary: str,
    source_english: str,
    *,
    min_overlap: float = 0.12,
) -> dict:
    """Flag Decision/Action lines that cannot be traced to the English source.

    Lightweight lexical overlap — not NLI. Soft-warn only; does not rewrite text.
    """
    summary = (summary or "").strip()
    source_english = (source_english or "").strip()
    if not summary or not source_english:
        return {"status": "skipped", "untraced": [], "checked": 0}

    src_words = _content_words(source_english)
    if not src_words:
        return {"status": "skipped", "untraced": [], "checked": 0}

    section = "Discussion"
    untraced: list[dict] = []
    checked = 0
    for raw in summary.splitlines():
        line = raw.strip()
        if not line:
            continue
        # Section headers from _format_meeting_minutes / topic blocks.
        header = line.rstrip(":").strip()
        if header in {"Discussion", "Decisions", "Action items"} and not line[:1].isdigit():
            if not line.startswith(("•", "-", "*")) and not re.match(r"^\d+\.", line):
                section = header
                continue
        body = re.sub(r"^([•\-\*]|\d+\.)\s*", "", line).strip()
        if not body:
            continue
        if section not in {"Decisions", "Action items"}:
            continue
        checked += 1
        words = _content_words(body)
        if not words:
            continue
        overlap = len(words & src_words) / len(words)
        if overlap < min_overlap:
            untraced.append(
                {
                    "section": section,
                    "line": body,
                    "overlap": round(overlap, 4),
                }
            )

    status = "warn" if untraced else "ok"
    return {"status": status, "untraced": untraced, "checked": checked}


def assess_translation_faithfulness(
    source: str,
    translation: str,
    *,
    min_overlap: float = 0.08,
    glossary: list[str] | None = None,
    review_lines: list[dict] | None = None,
) -> dict:
    """Flag likely MT errors and uncertain language-router lines.

    Lexical / glossary checks — not NLI. Also merges ``review_lines`` from the
    three-way EN/Tagalog/Hiligaynon router (hil↔tl ambiguity).
    """
    source = (source or "").strip()
    translation = (translation or "").strip()
    untraced: list[dict] = []
    checked = 0

    # Uncertain Hiligaynon/Tagalog/Cebuano disambiguation — always surface.
    for item in review_lines or []:
        line = (item.get("line") or "").strip()
        if not line:
            continue
        checked += 1
        untraced.append(
            {
                "section": item.get("section") or "Language review",
                "line": line[:180],
                "overlap": float(item.get("overlap") or 0.0),
            }
        )

    # Surface kept-source spans that were never machine-translated.
    for span in iter_untranslated_spans(translation):
        checked += 1
        untraced.append(
            {
                "section": "Untranslated",
                "line": span[:180],
                "overlap": 0.0,
            }
        )

    if not source or not translation:
        status = "warn" if untraced else "skipped"
        return {"status": status, "untraced": untraced, "checked": checked}

    # Glossary / sentence checks ignore untranslated wrappers (not English MT).
    translation_en = strip_untranslated_spans(translation, keep_inner=False)

    # Glossary terms in the source should survive unchanged in the translation.
    for term in glossary or []:
        if not term or term.casefold() not in source.casefold():
            continue
        checked += 1
        if (
            term not in translation_en
            and term.casefold() not in translation_en.casefold()
            and term not in translation
            and term.casefold() not in translation.casefold()
        ):
            untraced.append(
                {
                    "section": "Glossary",
                    "line": term,
                    "overlap": 0.0,
                }
            )

    # Sentence-level content overlap for longer units.
    src_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", source) if s.strip()]
    eng_words = _content_words(translation_en)
    if not eng_words:
        return {"status": "warn" if untraced else "skipped", "untraced": untraced, "checked": checked}

    for sent in src_sents[:40]:
        words = _content_words(sent)
        # Skip PH-only short fragments that won't overlap English lexically.
        if len(words) < 4:
            continue
        checked += 1
        # Proper-noun-ish tokens (capitalized) should often appear in EN.
        proper = {w for w in words if w[:1].isupper() and len(w) > 2}
        if proper:
            hit = len({p.casefold() for p in proper} & {e.casefold() for e in eng_words})
            overlap = hit / max(1, len(proper))
            if overlap < min_overlap and hit == 0:
                untraced.append(
                    {
                        "section": "Translation",
                        "line": sent[:180],
                        "overlap": round(overlap, 4),
                    }
                )

    status = "warn" if untraced else "ok"
    return {"status": status, "untraced": untraced, "checked": checked}


def _idea_preserving_summary(text: str) -> list[str]:
    """Primary path for short meeting speech: keep every distinct point."""
    return _segment_idea_units(text)


def _resolve_content_kind(source_kind: str | None) -> str:
    """Map ``source_kind`` to ``meeting`` | ``general`` content behavior."""
    kind = (source_kind or "meeting").strip().lower()
    if kind in {"general", "lyric", "song", "narrative", "other"}:
        return "general"
    # meeting | english_translation (alias) | default
    return "meeting"


def _bart_frame_prefix(content_kind: str) -> str:
    """BART input prefix — meeting-biased only for meeting content."""
    if _resolve_content_kind(content_kind) == "meeting":
        return "Board meeting discussion and decisions."
    return "Summarize the following."


def _bart_summarize_chunk(
    summarizer,
    chunk: str,
    *,
    content_kind: str = "meeting",
) -> str:
    words = max(1, len(chunk.split()))
    prefix = _bart_frame_prefix(content_kind)
    framed = f"{prefix} {chunk.strip()}"
    approx_tokens = max(20, int(words * 1.3))
    if words <= 150:
        max_len = min(240, max(90, approx_tokens))
        min_len = min(max_len - 5, max(36, int(approx_tokens * 0.48)))
    elif words <= 400:
        max_len = min(340, max(140, int(approx_tokens * 0.72)))
        min_len = min(max_len - 10, max(56, int(approx_tokens * 0.32)))
    else:
        max_len = 300
        min_len = 100
    with _pipelines.summarizer_infer_lock():
        out = summarizer(
            framed,
            max_length=max_len,
            min_length=min_len,
            do_sample=False,
            num_beams=6,
            truncation=True,
            no_repeat_ngram_size=3,
        )
    return out[0]["summary_text"].strip()


def _bart_summarize_topics(
    text: str,
    *,
    content_kind: str = "meeting",
) -> list[tuple[str, str]]:
    """Return ``[(topic_label, paragraph_summary), ...]`` via divide-and-conquer.

    Carries a short overlap tail from the previous topic into the next BART
    input so decisions that span topic boundaries keep discourse context.
    """
    summarizer = _pipelines.summarizer()
    tokenizer = getattr(summarizer, "tokenizer", None)
    topics = segment_transcript_topics(text, tokenizer=tokenizer)
    if not topics:
        return []
    kind = _resolve_content_kind(content_kind)
    results: list[tuple[str, str]] = []
    prev_tail = ""
    for idx, chunk in enumerate(topics, start=1):
        label = _topic_label(chunk, idx)
        input_chunk = f"{prev_tail} {chunk}".strip() if prev_tail else chunk
        try:
            summary = _bart_summarize_chunk(
                summarizer, input_chunk, content_kind=kind
            )
        except Exception as exc:
            logger.warning("BART failed on topic %s (%s); using extractive.", idx, exc)
            summary = " ".join(_segment_idea_units(chunk)[:5])
        if summary:
            results.append((label, summary))
        # Overlap: last 1–2 sentences of this topic for the next pass.
        sents = _split_sentences(chunk)
        if len(sents) >= 2:
            prev_tail = " ".join(sents[-2:])
        elif sents:
            prev_tail = sents[-1]
        else:
            prev_tail = ""
    return results


def _bart_summarize(text: str, *, content_kind: str = "meeting") -> str:
    """Backward-compatible summary string (joined topic summaries)."""
    parts = [
        summary
        for _, summary in _bart_summarize_topics(text, content_kind=content_kind)
    ]
    return " ".join(parts)


def _merge_missing_units(
    source_units: list[str],
    summary_units: list[str],
    *,
    min_overlap: float = 0.55,
) -> list[str]:
    """Restore omitted source ideas; dedupe overlap-boundary duplicates first."""
    # Discourse tails can cause the same point in topic N and N+1 summaries.
    summary_units = _dedupe_units(list(summary_units))
    summary_words = [_content_words(u) for u in summary_units]
    merged = list(summary_units)
    for unit in source_units:
        words = _content_words(unit)
        if not words:
            continue
        covered = any(
            (len(words & sw) / max(1, len(words))) >= min_overlap for sw in summary_words
        )
        if not covered:
            merged.append(unit)
    return _dedupe_units(merged)


def _contextual_bart_summary(
    text: str,
    source_units: list[str],
    *,
    output_format: str,
    engine_prefix: str,
    min_overlap: float = 0.5,
    content_kind: str = "meeting",
) -> tuple[str, str]:
    """Topic-aware BART + coverage restore against ``source_units``."""
    kind = _resolve_content_kind(content_kind)
    topic_parts = _bart_summarize_topics(text, content_kind=kind)
    topic_sections: list[tuple[str, list[str]]] = []
    all_units: list[str] = []
    for label, raw in topic_parts:
        units_i = _summary_sentences_to_units(raw)
        topic_sections.append((label, units_i))
        all_units.extend(units_i)
    units = _merge_missing_units(
        source_units, all_units, min_overlap=min_overlap
    )
    if kind == "meeting":
        return (
            _format_meeting_minutes(
                units, output_format, content_kind="meeting"
            ),
            f"{engine_prefix}-minutes",
        )
    # General: keep topic headings when multi-topic and coverage didn't append
    # extra flat units; otherwise a single flat bullet list.
    if len(topic_sections) > 1 and len(units) == len(all_units):
        return (
            _format_topic_summaries(topic_sections, output_format),
            f"{engine_prefix}-topic-chunks",
        )
    return _format_summary(units, output_format), f"{engine_prefix}-general"


def summarize(
    text: str,
    output_format: str = "bullets",
    source_kind: str = "meeting",
) -> tuple[str, str]:
    """Return (formatted_summary, engine_name).

    ``source_kind``:
    - ``meeting`` / ``english_translation`` (alias): topic-aware BART with
      meeting framing + Discussion / Decisions / Action items.
    - ``general``: topic-aware BART with neutral framing + flat/topic bullets
      (no minutes bucketing).
    - ``transcript``: raw (often PH) path — short extractive or topic BART
      without minutes bucketing; prefer ``summarize_to_english`` instead.
    """
    text = _normalize_spoken_transcript(text or "")
    if not text:
        return "", "none"

    source_units = _idea_preserving_summary(text)
    word_count = len(text.split())
    kind_raw = (source_kind or "meeting").strip().lower()
    if kind_raw == "english_translation":
        kind_raw = "meeting"
    content_kind = _resolve_content_kind(kind_raw)

    # Already-English paths (post-MT or explicit meeting/general).
    if kind_raw in {"meeting", "general"}:
        units = source_units
        engine = f"bart-from-{content_kind}"
        if word_count >= 24 and len(source_units) >= 2:
            try:
                return _contextual_bart_summary(
                    text,
                    source_units,
                    output_format=output_format,
                    engine_prefix=f"bart-{content_kind}",
                    min_overlap=0.42,
                    content_kind=content_kind,
                )
            except Exception as exc:
                logger.warning(
                    "Topic-aware BART (%s) failed (%s); keeping idea units.",
                    content_kind,
                    exc,
                )
                units = source_units
        if content_kind == "meeting":
            return (
                _format_meeting_minutes(
                    units, output_format, content_kind="meeting"
                ),
                engine,
            )
        return _format_summary(units, output_format), engine

    # Raw (often PH) transcript path — prefer translating first via
    # ``summarize_to_english``. Kept for direct/legacy callers.
    if word_count <= 220 or len(source_units) <= 8:
        return _format_summary(source_units, output_format), "bart-meeting-minutes"

    try:
        return _contextual_bart_summary(
            text,
            source_units,
            output_format=output_format,
            engine_prefix="bart",
            min_overlap=0.55,
            content_kind="general",
        )
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


def _english_covers_transcript(english: str, transcript: str) -> bool:
    """True when cached English is long enough and content-faithful."""
    english = (english or "").strip()
    transcript = (transcript or "").strip()
    en_words = len(english.split())
    src_words = len(transcript.split())
    if en_words < 8 or not _looks_like_latin_script(english):
        return False
    if src_words <= 0:
        return True
    # Allow compression, but reject stubs that dropped most of the meeting.
    if en_words < max(8, int(src_words * 0.35)):
        return False
    src_en, src_fi = _language_scores(transcript)
    # Mostly-English source: require content-word overlap (reuse dead helper).
    if src_fi < 0.08 and src_en >= src_fi:
        return _coverage_ratio(transcript, english) >= 0.22
    # PH / mixed source: length + Latin script is the practical gate
    # (content words often change completely after translation).
    return True


def summarize_to_english(
    transcript: str,
    *,
    source_language: str = "auto",
    output_format: str = "bullets",
    existing_english: str | None = None,
    source_kind: str = "meeting",
) -> tuple[str, str, str, str, list[dict]]:
    """Translate the full transcript to English, then summarize (BART).

    ``source_kind`` is ``meeting`` (default) or ``general`` — controls BART
    framing and whether Discussion / Decisions / Action items bucketing runs.

    Returns
    ``(summary, summary_engine, english_text, translate_engine, mt_review_lines)``.
    ``mt_review_lines`` are uncertain Hiligaynon/Tagalog router flags for manual review.
    """
    source = _normalize_spoken_transcript(transcript or "")
    if not source:
        return "", "none", "", "none", []

    content_kind = _resolve_content_kind(source_kind)
    english = (existing_english or "").strip()
    translate_engine = "cached-english"
    mt_review: list[dict] = []
    if not _english_covers_transcript(english, source):
        tr = translate(
            source,
            target_language="en",
            source_language=source_language or "auto",
        )
        english = _normalize_spoken_transcript(tr.text or "")
        translate_engine = tr.engine
        mt_review = list(tr.review_lines or [])
        if not english:
            summary, engine = summarize(
                source, output_format=output_format, source_kind="transcript"
            )
            return summary, engine, source, translate_engine or "none", mt_review

    # BART must not see raw kept-source Tagalog wrapped as [untranslated: …].
    english_for_summary = strip_untranslated_spans(english, keep_inner=False)
    if not english_for_summary.strip():
        english_for_summary = strip_untranslated_spans(english, keep_inner=True)

    summary, summary_engine = summarize(
        english_for_summary,
        output_format=output_format,
        source_kind=content_kind,
    )
    return summary, summary_engine, english, translate_engine, mt_review


def _nllb_src_code(source_language: str, text: str = "") -> str:
    """Map app/meeting language to an NLLB source code for PH → English."""
    primary = (source_language or "auto").strip().lower()
    if primary in {"hil", "hiligaynon", "ilonggo"}:
        # NLLB has no Hiligaynon code — ceb_Latn is an emergency Visayan proxy
        # (same structural gap as mBART's missing hil_XX; Google is preferred).
        return "ceb_Latn"
    if primary in {"tl", "fil", "filipino", "tagalog"}:
        return "tgl_Latn"
    if primary in {"id", "indonesian"}:
        return "ind_Latn"
    # auto / mixed: prefer Tagalog when Filipino markers dominate.
    _, fi = _language_scores(text or "")
    if fi >= 0.08:
        return "tgl_Latn"
    return "tgl_Latn"


def _nllb_translate_to_english(text: str, source_language: str = "auto") -> str:
    """Translate Philippine / mixed text to English with NLLB (real Tagalog)."""
    model, tokenizer = _pipelines.nllb()
    src = _nllb_src_code(source_language, text)
    # Transformers NLLB tokenizers expose lang code → id via convert_tokens / lang_code_to_id.
    bos_id = None
    if hasattr(tokenizer, "lang_code_to_id"):
        bos_id = tokenizer.lang_code_to_id.get("eng_Latn")
    if bos_id is None:
        try:
            bos_id = tokenizer.convert_tokens_to_ids("eng_Latn")
        except Exception:
            bos_id = None
    if bos_id is None:
        raise LLMUnavailable("NLLB tokenizer missing eng_Latn")

    outputs: list[str] = []
    with _pipelines.nllb_infer_lock():
        if hasattr(tokenizer, "src_lang"):
            tokenizer.src_lang = src
        for chunk in _chunk_text(
            text, overlap_chars=200 if len(text) > _MAX_CHUNK_CHARS else 0
        ):
            encoded = tokenizer(
                chunk, return_tensors="pt", truncation=True, max_length=512
            )
            generated = model.generate(
                **encoded,
                forced_bos_token_id=bos_id,
                max_new_tokens=min(512, max(64, int(len(chunk.split()) * 2.5))),
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=4,
            )
            cleaned = tokenizer.batch_decode(
                generated, skip_special_tokens=True
            )[0].strip()
            cleaned = _collapse_translation_loops(cleaned)
            if not _looks_like_latin_script(cleaned):
                raise _NonEnglishTranslation(cleaned)
            if _is_garbage_english_translation(chunk, cleaned):
                raise _NonEnglishTranslation(cleaned)
            outputs.append(cleaned)
    return " ".join(outputs).strip()


def _mbart_translate(text: str, src_code: str, tgt_code: str) -> str:
    model, tokenizer = _pipelines.mbart()
    src = mbart_code(src_code)
    if not src:
        # Historical bug: ``or "en_XX"`` mistagged unmapped PH codes as English
        # with no log line — looked like "translation didn't happen".
        logger.warning(
            "Unmapped mBART source code %r — refusing silent English mistag; "
            "using id_ID as emergency typological proxy. Add the code to "
            "languages.py::LANGUAGES / aliases.",
            src_code,
        )
        src = "id_ID"
    tgt = mbart_code(tgt_code)
    if not tgt:
        logger.warning("Unmapped mBART target code %r", tgt_code)
        raise LLMUnavailable(f"Unsupported target language: {tgt_code}")

    # Identity English→English is unreliable on this checkpoint (emits te_IN etc.).
    if src == "en_XX" and tgt == "en_XX":
        return text.strip()

    bos_id = tokenizer.lang_code_to_id.get(tgt)
    if bos_id is None:
        bos_id = tokenizer.convert_tokens_to_ids(tgt)

    outputs: list[str] = []
    # Serialize tokenize+generate — Fast tokenizers raise "Already borrowed"
    # when used concurrently from summarize/translate auto-paths.
    with _pipelines.mbart_infer_lock():
        # Overlap long inputs so discourse across chunk edges is not lost.
        for chunk in _chunk_text(
            text, overlap_chars=280 if len(text) > _MAX_CHUNK_CHARS else 0
        ):
            tokenizer.src_lang = src
            encoded = tokenizer(
                chunk, return_tensors="pt", truncation=True, max_length=1024
            )
            generated = model.generate(
                **encoded,
                forced_bos_token_id=bos_id,
                max_new_tokens=min(512, max(64, int(len(chunk.split()) * 2.8))),
                num_beams=6,
                early_stopping=True,
                no_repeat_ngram_size=4,
                length_penalty=1.05,
            )
            raw = tokenizer.batch_decode(generated, skip_special_tokens=False)[0]
            # Drop any leaked language-code tokens the decoder may prepend after BOS.
            cleaned = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
            cleaned = _strip_leaked_lang_codes(cleaned)
            cleaned = _collapse_translation_loops(cleaned)
            if tgt == "en_XX" and not _looks_like_latin_script(cleaned):
                logger.warning(
                    "mBART %s→%s produced non-Latin output (%s…); will retry upstream.",
                    src,
                    tgt,
                    cleaned[:40],
                )
                logger.debug("Raw decode: %s", raw[:120])
                raise _NonEnglishTranslation(cleaned)
            if tgt == "en_XX" and _is_garbage_english_translation(chunk, cleaned):
                logger.warning(
                    "mBART %s→en produced word-salad (%s…); will retry upstream.",
                    src,
                    cleaned[:48],
                )
                raise _NonEnglishTranslation(cleaned)
            outputs.append(cleaned)
    return " ".join(outputs).strip()


class _NonEnglishTranslation(RuntimeError):
    """Internal signal that mBART failed to emit English script."""


_LANG_CODE_LEAK_RE = re.compile(
    r"^(?:en_XX|es_XX|fr_XX|de_DE|it_IT|pt_XX|ar_AR|hi_IN|ja_XX|zh_CN|ru_RU|"
    r"nl_XX|ko_KR|id_ID|tl_XX|ps_AF|te_IN|ur_PK|fa_IR)+\s*",
    re.IGNORECASE,
)
_REPEAT_PHRASE_RE = re.compile(r"\b(.{2,40}?)(?:\s+\1){2,}\b", re.IGNORECASE)
_REPEAT_SENTENCE_RE = re.compile(
    r"([^.!?]{8,120}[.!?])(?:\s+\1){2,}",
    re.IGNORECASE,
)


def _strip_leaked_lang_codes(text: str) -> str:
    return _LANG_CODE_LEAK_RE.sub("", text).strip()


def _collapse_translation_loops(text: str) -> str:
    prev = None
    text = (text or "").strip()
    while prev != text:
        prev = text
        text = _REPEAT_PHRASE_RE.sub(r"\1", text)
        text = _REPEAT_SENTENCE_RE.sub(r"\1", text)
        text = _WS_RE.sub(" ", text).strip()
    return text


def _is_garbage_english_translation(source: str, translated: str) -> bool:
    """Detect mBART/NLLB word-salad and broken English on PH→EN.

    Combines the historical salad-word list with vocabulary-free checks
    (length ratio, stray single-letter tokens) so failures like
    ``sa mga halang…`` → ``To honor m Swear`` are caught too.
    """
    dst = _collapse_translation_loops(translated or "")
    if not dst:
        return True
    src_tokens = re.findall(r"[A-Za-zÀ-ÿ']+", (source or "").lower())
    dst_tokens = re.findall(r"[A-Za-zÀ-ÿ']+", dst.lower())
    if len(dst_tokens) < 2:
        return True
    # Heavy repetition remaining after collapse.
    if len(dst_tokens) >= 8:
        uniq = len(set(dst_tokens)) / float(len(dst_tokens))
        if uniq < 0.35:
            return True
    # Collapse removed most of a long translation → was a loop.
    raw_n = len(re.findall(r"[A-Za-zÀ-ÿ']+", (translated or "").lower()))
    if raw_n >= 20 and len(dst_tokens) < max(6, int(raw_n * 0.4)):
        return True
    # Nonsense short-noun salad unrelated to source (classic id_ID failure mode).
    salad = {
        "bow", "arrow", "rope", "rail", "wing", "stone", "tail", "stub",
        "row", "rows", "wing", "wings",
    }
    if len(dst_tokens) >= 8:
        salad_hits = sum(1 for t in dst_tokens if t in salad)
        if salad_hits / len(dst_tokens) >= 0.35:
            return True
    # Stray single-letter tokens (except a/I) → broken fragments like "honor m Swear".
    stray_letters = [
        t for t in dst_tokens if len(t) == 1 and t.lower() not in {"a", "i"}
    ]
    if stray_letters and len(dst_tokens) <= 16:
        return True
    # Implausible length vs source (PH→EN is rarely <35% or >3.5× token count).
    if len(src_tokens) >= 4:
        ratio = len(dst_tokens) / float(len(src_tokens))
        if ratio < 0.35 or ratio > 3.5:
            return True
    # Source was clearly Filipino but output still has many Filipino markers.
    if src_tokens:
        fi = sum(1 for t in src_tokens if t in _FILIPINO_MARKERS) / max(1, len(src_tokens))
        fi_out = sum(1 for t in dst_tokens if t in _FILIPINO_MARKERS) / max(
            1, len(dst_tokens)
        )
        if fi >= 0.12 and fi_out >= 0.12 and fi_out >= fi * 0.6:
            return True
    return False


def _looks_like_latin_script(text: str) -> bool:
    """True when the text is predominantly Latin letters (English output check)."""
    if not text or not text.strip():
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    latin = sum(1 for c in letters if ("a" <= c.lower() <= "z") or ("A" <= c <= "Z"))
    return (latin / len(letters)) >= 0.75


def _is_mostly_english_sentence(sentence: str) -> bool:
    """True only when the clause is clearly English (mixed PH must translate)."""
    en, fi = _language_scores(sentence)
    # Any meaningful Filipino signal → translate (code-switched board speech).
    if fi >= 0.08:
        return False
    if _FILIPINO_STARTER_RE.search(f"x {sentence}"):
        return False
    tokens = re.findall(r"[A-Za-zÀ-ÿ']+", sentence)
    if not tokens:
        return _looks_like_latin_script(sentence)
    # Require English markers to dominate; ambiguous Latin text still translates
    # when the meeting language is PH (handled by callers via src candidates).
    return en >= fi and en >= 0.06


# Context-window MT: short units skip neighbors (span trim is unreliable there).
_CONTEXT_MIN_UNIT_WORDS = 6
# Target/source sentence-count ratio must stay in this band to trust trimming.
_CONTEXT_SENTENCE_PARITY_MIN = 0.5
_CONTEXT_SENTENCE_PARITY_MAX = 2.0
# Extracted span must cover at least this fraction of the unit's word count.
_CONTEXT_SPAN_MIN_WORD_RATIO = 0.35


def _word_count(text: str) -> int:
    return len((text or "").split())


def _extract_target_span(full_translation: str, unit: str, window: str) -> str:
    """From a context-window translation, keep the span for ``unit``.

    Uses a source word-count ratio as a sentence-count heuristic on the target.
    Callers must validate sentence-count parity and span length before trusting
    the result — Tagalog/Hiligaynon → English often changes sentence boundaries.
    """
    full = (full_translation or "").strip()
    if not full:
        return ""
    sentences = _split_sentences(full)
    if len(sentences) <= 1:
        return full
    unit_ratio = _word_count(unit) / max(1, _word_count(window))
    keep = max(1, min(len(sentences), int(math.ceil(len(sentences) * unit_ratio))))
    # Prefer trailing sentences (context was prefixed).
    return " ".join(sentences[-keep:]).strip()


def _span_looks_truncated(span: str, unit: str) -> bool:
    """True when ``span`` is empty or suspiciously short vs the source unit."""
    span = (span or "").strip()
    if not span:
        return True
    unit_words = _word_count(unit)
    if unit_words <= 0:
        return False
    min_words = max(1, int(math.ceil(unit_words * _CONTEXT_SPAN_MIN_WORD_RATIO)))
    return _word_count(span) < min_words


def _sentence_count_parity_ok(window: str, translation: str) -> bool:
    """True when target sentence count is roughly aligned with the source window."""
    src_n = max(1, len(_split_sentences(window)))
    tgt_n = max(1, len(_split_sentences(translation)))
    ratio = tgt_n / src_n
    return _CONTEXT_SENTENCE_PARITY_MIN <= ratio <= _CONTEXT_SENTENCE_PARITY_MAX


def _translate_unit_alone(unit: str, src: str, *, engine: str) -> str:
    """Context-free unit translation (NLLB or mBART). Google is handled upstream."""
    if engine == "nllb":
        return _nllb_translate_to_english(unit, source_language=src)
    return _mbart_translate(unit, src, "en")


def _translate_unit_with_context(
    unit: str,
    prev_units: list[str],
    src: str,
    *,
    context_n: int = 2,
    engine: str = "mbart",
) -> str:
    """Translate one unit with preceding units as discourse context.

    Falls back to a context-free translation when the unit is short, when
    source/target sentence counts diverge, or when span extraction would drop
    or leak content. Hiligaynon Google Translate always stays line-local.
    """
    unit = (unit or "").strip()
    if not unit:
        return ""

    if engine == "google":
        from . import google_translate

        # Line-local: avoid mixing EN/Tagalog context into Hiligaynon Google calls.
        return google_translate.translate_hiligaynon_to_english(unit)

    unit_words = _word_count(unit)
    if unit_words < _CONTEXT_MIN_UNIT_WORDS:
        logger.info(
            "Context-window skipped (short unit, words=%d < %d): %s…",
            unit_words,
            _CONTEXT_MIN_UNIT_WORDS,
            unit[:48],
        )
        return _translate_unit_alone(unit, src, engine=engine)

    ctx = [u.strip() for u in prev_units[-context_n:] if (u or "").strip()]
    # Drop untranslated markers from discourse context (they're not English MT).
    ctx = [strip_untranslated_spans(c, keep_inner=True) for c in ctx]
    ctx = [c for c in ctx if c]
    if not ctx:
        return _translate_unit_alone(unit, src, engine=engine)

    # EN preamble + PH lyric/verse confuses NLLB/mBART and truncates the verse
    # (observed: content after "saksihin" silently dropped). Skip cross-lang context.
    ctx_en, ctx_fi = _language_scores(" ".join(ctx))
    unit_en, unit_fi = _language_scores(unit)
    cross_lang = (ctx_en > ctx_fi + 0.04 and unit_fi > unit_en + 0.04) or (
        ctx_fi > ctx_en + 0.04 and unit_en > unit_fi + 0.04
    )
    if cross_lang:
        logger.info(
            "Context-window skipped (cross-language context en=%.2f/fi=%.2f "
            "vs unit en=%.2f/fi=%.2f): %s…",
            ctx_en,
            ctx_fi,
            unit_en,
            unit_fi,
            unit[:48],
        )
        return _translate_unit_alone(unit, src, engine=engine)

    window = " ".join(ctx + [unit])
    if engine == "nllb":
        full = _nllb_translate_to_english(window, source_language=src)
    else:
        full = _mbart_translate(window, src, "en")
    full = (full or "").strip()
    if not full:
        logger.info(
            "Context-window discarded (empty window translation); "
            "retrying unit alone: %s…",
            unit[:48],
        )
        return _translate_unit_alone(unit, src, engine=engine)

    src_n = max(1, len(_split_sentences(window)))
    tgt_n = max(1, len(_split_sentences(full)))
    if not _sentence_count_parity_ok(window, full):
        logger.info(
            "Context-window discarded (sentence parity src=%d tgt=%d ratio=%.2f); "
            "retrying unit alone: %s…",
            src_n,
            tgt_n,
            tgt_n / src_n,
            unit[:48],
        )
        return _translate_unit_alone(unit, src, engine=engine)

    span = _extract_target_span(full, unit, window)
    if not (span or "").strip():
        logger.info(
            "Context-window discarded (empty span); retrying unit alone: %s…",
            unit[:48],
        )
        return _translate_unit_alone(unit, src, engine=engine)
    if _span_looks_truncated(span, unit):
        logger.info(
            "Context-window discarded (truncated span, span_words=%d unit_words=%d); "
            "retrying unit alone: %s…",
            _word_count(span),
            unit_words,
            unit[:48],
        )
        return _translate_unit_alone(unit, src, engine=engine)
    if _is_garbage_english_translation(unit, span):
        logger.info(
            "Context-window discarded (garbage span); retrying unit alone: %s…",
            unit[:48],
        )
        return _translate_unit_alone(unit, src, engine=engine)
    return span.strip()


@dataclass
class TranslateResult:
    """English translation plus three-way router metadata."""

    text: str
    engine: str
    review_lines: list[dict] = field(default_factory=list)
    route_counts: dict = field(default_factory=dict)

    def __iter__(self):
        yield self.text
        yield self.engine


def _route_attempts_for_line(
    route_lang: str,
    *,
    prefer_mbart: bool,
    has_ph_mbart: bool,
    use_nllb: bool,
) -> list[tuple[str, str]]:
    """Build ordered (engine, src) attempts for one routed line.

    Tagalog mBART attempts always pass shorthand ``tl`` (native ``tl_XX``).
    The old ``id`` substitution for stock mBART was an inherited workaround;
    fixture benchmark prefers ``tl_XX`` over ``id_ID`` (see docs/MBART_PH_AUDIT.md).

    Hiligaynon has no stock mBART token — Google first, NLLB ``ceb_Latn`` next,
    then mBART as a **degraded** last resort. When a PH fine-tune is configured,
    the last-resort shorthand is ``hil`` so ``mbart_code`` can resolve to a real
    ``hil_XX`` (vocab-extended checkpoint) or legacy ``tl_XX`` proxy. Without a
    fine-tune it stays ``id`` (typological proxy). ``has_ph_mbart`` only affects
    that Hiligaynon last-resort tag, not Tagalog.
    """
    from . import google_translate

    attempts: list[tuple[str, str]] = []
    if route_lang == "hil":
        # Primary: Google ``hil``. mBART is never a peer option here.
        if google_translate.is_configured():
            attempts.append(("google", "hil"))
        fallback = (settings.hil_translate_fallback or "nllb").strip().lower()
        if fallback == "nllb" and use_nllb:
            attempts.append(("nllb", "hil"))
        # Last resort: ``hil`` → hil_XX or legacy tl_XX via mbart_code(); else id_ID.
        attempts.append(("mbart", "hil" if has_ph_mbart else "id"))
        return attempts
    if route_lang == "tl":
        # Always native Tagalog token via shorthand ``tl`` → ``tl_XX``.
        if prefer_mbart:
            attempts.append(("mbart", "tl"))
        if use_nllb:
            attempts.append(("nllb", "tl"))
        if not prefer_mbart:
            attempts.append(("mbart", "tl"))
        attempts.append(("mbart", "en"))
        return attempts
    if use_nllb:
        attempts.append(("nllb", "auto"))
    # Unknown / mixed: try Tagalog-native mBART before English identity.
    attempts.append(("mbart", "tl"))
    attempts.append(("mbart", "en"))
    return attempts


def _translate_to_english(text: str, source_language: str) -> TranslateResult:
    """Three-way translate: EN passthrough / Tagalog→NLLB / Hiligaynon→Google.

    1. Collapse Whisper loops, split into idea units (order preserved).
    2. Classify each unit with :mod:`lang_router` (Hiligaynon wordlist — not
       langdetect, which mislabels Ilonggo as Tagalog/Cebuano).
    3. Route: English unchanged; Tagalog → NLLB ``tgl_Latn``; Hiligaynon →
       Google Cloud Translation ``hil`` (NLLB ceb_Latn only as fallback).
    4. Reassemble in original unit order; flag uncertain hil/tl lines for review.
    """
    from . import lang_router

    normalized = _normalize_spoken_transcript(text or "")
    if not normalized:
        return TranslateResult("", "none")
    try:
        from .transcription import _collapse_hallucinations

        normalized = _collapse_hallucinations(normalized)
    except Exception:
        normalized = _collapse_translation_loops(normalized)
    if not normalized:
        return TranslateResult("", "none")

    units = _segment_idea_units(normalized)
    if len(units) < 2:
        units = _split_sentences(_WS_RE.sub(" ", normalized).strip())
    if not units:
        return TranslateResult("", "none")

    primary = (source_language or "auto").strip().lower()
    backend = (settings.ph_translate_backend or "auto").strip().lower()
    has_ph_mbart = bool((settings.mbart_ph_finetuned_model or "").strip())
    prefer_mbart = backend == "mbart" or (backend == "auto" and has_ph_mbart)
    use_nllb = backend != "mbart"

    out_units: list[str] = []
    review_lines: list[dict] = []
    route_counts = {"en": 0, "tl": 0, "hil": 0, "unknown": 0, "uncertain": 0}
    engines_used: set[str] = set()
    kept_source = 0
    ph_source_mass = 0
    ph_translated_mass = 0

    routed = lang_router.route_units(units, meeting_language=primary)
    for i, (unit, decision) in enumerate(routed):
        try:
            from .transcription import _collapse_hallucinations, _is_junk_transcript

            unit_clean = _collapse_hallucinations(unit)
            if not unit_clean or _is_junk_transcript(unit_clean):
                continue
            unit = unit_clean
            decision = lang_router.classify_line(unit, meeting_language=primary)
        except Exception:
            pass

        route_lang = decision.language
        if route_lang == "en" or (
            route_lang == "unknown" and _is_mostly_english_sentence(unit)
        ):
            out_units.append(unit)
            route_counts["en"] += 1
            engines_used.add("passthrough-english")
            if decision.uncertain:
                route_counts["uncertain"] += 1
                review_lines.append(
                    {
                        "section": "Language review",
                        "line": unit[:180],
                        "overlap": round(decision.confidence, 4),
                    }
                )
            continue

        if route_lang not in {"en", "tl", "hil", "unknown"}:
            route_lang = "unknown"
        route_counts[route_lang] += 1
        if decision.uncertain or route_lang == "unknown":
            route_counts["uncertain"] += 1
            review_lines.append(
                {
                    "section": "Language review",
                    "line": unit[:180],
                    "overlap": round(decision.confidence, 4),
                }
            )

        unit_mass = _token_mass(unit)
        ph_source_mass += unit_mass

        attempts = _route_attempts_for_line(
            route_lang,
            prefer_mbart=prefer_mbart,
            has_ph_mbart=has_ph_mbart,
            use_nllb=use_nllb,
        )
        piece = ""
        last_err = None
        for engine, src in attempts:
            try:
                piece = _translate_unit_with_context(
                    unit,
                    out_units if out_units else units[:i],
                    src,
                    engine=engine,
                )
                if not _looks_like_latin_script(piece):
                    raise _NonEnglishTranslation(piece)
                if _is_garbage_english_translation(unit, piece):
                    raise _NonEnglishTranslation(piece)
                if engine == "google":
                    engines_used.add("google-translate-hil")
                elif engine == "nllb":
                    engines_used.add("nllb-200")
                elif has_ph_mbart and src == "tl":
                    engines_used.add("mbart-ph-finetuned")
                else:
                    engines_used.add("mbart-large-50")
                break
            except _NonEnglishTranslation as exc:
                last_err = exc
                piece = ""
                continue
            except Exception as exc:
                last_err = exc
                piece = ""
                logger.debug(
                    "Unit translate engine=%s src=%s route=%s failed: %s",
                    engine,
                    src,
                    route_lang,
                    exc,
                )
                continue
        if not piece:
            logger.warning(
                "Keeping source clause after translation failures "
                "(route=%s last=%s): %s…",
                route_lang,
                last_err,
                unit[:48],
            )
            # Visible marker — do not silently splice Tagalog into "English".
            piece = mark_untranslated(unit)
            kept_source += 1
            # Untranslated mass does not count toward English coverage.
        else:
            ph_translated_mass += _token_mass(piece)
        out_units.append(piece)

    joined = " ".join(s for s in out_units if s).strip()
    # Collapse EN loops without rewriting [untranslated: …] spans.
    rebuilt: list[str] = []
    cursor = 0
    for match in _UNTRANSLATED_RE.finditer(joined):
        head = joined[cursor : match.start()]
        if head.strip():
            rebuilt.append(_collapse_translation_loops(head))
        rebuilt.append(match.group(0))
        cursor = match.end()
    tail = joined[cursor:]
    if tail.strip():
        rebuilt.append(_collapse_translation_loops(tail))
    joined = " ".join(rebuilt).strip() if rebuilt else _collapse_translation_loops(joined)

    if not engines_used:
        engine_name = "passthrough-english"
    elif len(engines_used) == 1:
        engine_name = next(iter(engines_used))
    else:
        order = [
            "google-translate-hil",
            "nllb-200",
            "mbart-ph-finetuned",
            "mbart-large-50",
            "passthrough-english",
        ]
        engine_name = "+".join(e for e in order if e in engines_used)

    if kept_source >= 1:
        logger.info(
            "Three-way MT kept %d source clauses (marked [untranslated:…]); "
            "routes=%s uncertain=%d",
            kept_source,
            {k: v for k, v in route_counts.items() if k != "uncertain"},
            route_counts["uncertain"],
        )
        review_lines.append(
            {
                "section": "Untranslated",
                "line": f"{kept_source} clause(s) kept as source after MT failure",
                "overlap": 0.0,
            }
        )

    # Issue 5: silent content-loss detector on PH→EN token mass.
    mass_cov = _translation_mass_coverage(ph_source_mass, ph_translated_mass)
    if ph_source_mass >= 8 and mass_cov < _TRANSLATION_MASS_COVERAGE_MIN:
        logger.warning(
            "Translation mass coverage low: %.2f (ph_src_tokens=%d ph_dst_tokens=%d) "
            "— possible silent content loss",
            mass_cov,
            ph_source_mass,
            ph_translated_mass,
        )
        review_lines.append(
            {
                "section": "Coverage",
                "line": (
                    f"PH→EN token-mass coverage {mass_cov:.0%} "
                    f"({ph_translated_mass}/{ph_source_mass}) below "
                    f"{_TRANSLATION_MASS_COVERAGE_MIN:.0%}"
                ),
                "overlap": round(mass_cov, 4),
            }
        )

    return TranslateResult(
        text=joined,
        engine=engine_name,
        review_lines=review_lines,
        route_counts=route_counts,
    )


def translate(
    text: str, target_language: str, source_language: str = "auto"
) -> TranslateResult:
    text = (text or "").strip()
    if not text:
        return TranslateResult("", "none")
    tgt = (target_language or "en").strip().lower()
    if tgt == "en":
        return _translate_to_english(text, source_language)
    src = (source_language or "auto").strip().lower()
    if src in {"auto", "detect", "none", ""}:
        _, fi = _language_scores(text)
        src = "id" if fi >= 0.08 else "en"
    return TranslateResult(_mbart_translate(text, src, tgt), "mbart-large-50")



def invoke_llm(task: str, text: str, **kwargs):
    if task == "summarize":
        return summarize(
            text,
            output_format=kwargs.get("output_format", "bullets"),
            source_kind=kwargs.get("source_kind", "meeting"),
        )
    if task == "translate":
        return translate(
            text,
            target_language=kwargs["target_language"],
            source_language=kwargs.get("source_language", "auto"),
        )
    raise ValueError(f"Unknown InvokeLLM task: {task}")


def summarizer_available() -> bool:
    try:
        import transformers  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False
