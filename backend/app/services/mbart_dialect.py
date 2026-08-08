"""mBART dialect normalizer (SmartScribe-style system prompt → pre-MT cleanup).

``facebook/mbart-large-50`` is a seq2seq MT model — it does **not** accept chat
system prompts. This module stores the SmartScribe ``MBART_SYSTEM_PROMPT`` and
**operationalizes its rules** as deterministic cleanup before mBART encode:

1. Remove fillers / false starts
2. Preserve proper nouns, numbers, glossary placeholders
3. Light Taglish cleanup for stable ``tl_XX`` input
4. Label clear decision / action-item clauses (NAPAGPASYAHAN / AKSYON)

Cleaner formal Filipino/Tagalog input → more reliable mBART → English.
"""
from __future__ import annotations

import logging
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Iterator

logger = logging.getLogger(__name__)

# SmartScribe / Smart Meeting dialect-normalizer prompt (documentation + render).
# Not sent to mBART weights — rules are applied in ``normalize_for_mbart``.
MBART_SYSTEM_PROMPT = """
You are a dialect normalizer for the SmartScribe meeting documentation system.

Your task: Given raw spoken Hiligaynon or Filipino text transcribed by Whisper ASR,
normalize it into clear, formal written Filipino suitable for meeting minutes.

Rules:
- Preserve all proper nouns, names, and numbers exactly as spoken
- Convert code-switched English fragments to formal Filipino equivalents
- Remove filler words (uh, um, ah, 'di ba, kuan) and false starts
- Format decisions as: NAPAGPASYAHAN: [decision]
- Format action items as: AKSYON: [person] – [task] – [deadline]
- Do not add information not present in the original speech

Input language: {source_lang}
Output language: Filipino (formal)
Context: Meeting minutes for {meeting_title}
Participants: {participants}
"""


@dataclass
class MeetingContext:
    source_lang: str = "auto"
    meeting_title: str = "Untitled meeting"
    participants: list[str] = field(default_factory=list)

    def participants_str(self) -> str:
        if not self.participants:
            return "(unspecified)"
        return ", ".join(p for p in self.participants if p)


_CTX: ContextVar[MeetingContext | None] = ContextVar("mbart_meeting_ctx", default=None)

_WS_RE = re.compile(r"\s+")
# Glossary placeholders from glossary.protect — never alter.
_GLOSSARY_PH_RE = re.compile(r"⟦SMG\d+⟧")
# Spoken fillers (Hiligaynon / Filipino / English).
_FILLER_RE = re.compile(
    r"(?:^|\s+)(?:uh+|um+|ah+|eh+|uhm+|mm+|hmm+|'di\s*ba|di\s*ba|diba|"
    r"kuan|ano\s+ba|parang\s+ano|you\s+know|i\s+mean|like)\b[,.]?",
    re.IGNORECASE,
)
# False starts: repeated short word then continuation ("ang ang budget").
_FALSE_START_RE = re.compile(
    r"\b([A-Za-zÀ-ÿ']{1,12})\s+\1\b",
    re.IGNORECASE,
)
# Trailing ellipsis spam from Whisper.
_ELLIPSIS_RE = re.compile(r"(\s*\.\.\.\s*){2,}")

# Light Taglish → formal Filipino (only high-confidence meeting verbs).
# Applied before mBART so ``tl_XX`` sees more consistent Filipino.
_TAGLISH_MAP = (
    (re.compile(r"\bi-move\b", re.I), "ilipat"),
    (re.compile(r"\bi-record\b", re.I), "irekord"),
    (re.compile(r"\bi-approve\b", re.I), "aprubahan"),
    (re.compile(r"\bi-reject\b", re.I), "tanggihan"),
    (re.compile(r"\bi-presenta\b", re.I), "ipakita"),
    (re.compile(r"\bi-submit\b", re.I), "isumite"),
    (re.compile(r"\bi-assign\b", re.I), "italaga"),
    (re.compile(r"\bi-defer\b", re.I), "ipagpaliban"),
    (re.compile(r"\bi-table\b", re.I), "itabi muna"),
    (re.compile(r"\bi-circulate\b", re.I), "ipamahagi"),
    (re.compile(r"\bi-note\b", re.I), "itala"),
    (re.compile(r"\bi-call to order\b", re.I), "simulan ang meeting"),
    (re.compile(r"\bfollow-?up\b", re.I), "pagsunod"),
    (re.compile(r"\bdeadline\b", re.I), "takdang petsa"),
    (re.compile(r"\baction item\b", re.I), "aksyon"),
)

# Only *decided* outcomes — not future intent ("kailangan nating aprubahan…").
_DECISION_CUE_RE = re.compile(
    r"\b(?:napagpasyahan|napagkasunduan|naaprubahan|inaprubahan|"
    r"naaprubahan ang|motion (?:is )?approved|resolusyon ay naaprubahan)\b",
    re.IGNORECASE,
)
_ACTION_CUE_RE = re.compile(
    r"\b(?:aksyon|action item|responsable|itatagal|italaga|gagawin|"
    r"susundan|follow-?up|i-assign|assign)\b",
    re.IGNORECASE,
)
_PERSON_RE = re.compile(
    r"\b(?:kay|ni|si)\s+([A-ZÀ-Ý][A-Za-zÀ-ÿ'.\-]{1,40})\b"
)
_DEADLINE_RE = re.compile(
    r"\b(?:bago ang|before|sa|on)\s+"
    r"((?:susunod na linggo|next week|Lunes|Martes|Miyerkules|Huwebes|"
    r"Biyernes|Sabado|Linggo|Monday|Tuesday|Wednesday|Thursday|Friday|"
    r"deadline|takdang petsa)[^.!]*)",
    re.IGNORECASE,
)


def render_system_prompt(
    *,
    source_lang: str = "auto",
    meeting_title: str = "Untitled meeting",
    participants: list[str] | None = None,
) -> str:
    """Fill the SmartScribe prompt template (for logs / UI / docs)."""
    ctx = MeetingContext(
        source_lang=source_lang or "auto",
        meeting_title=meeting_title or "Untitled meeting",
        participants=list(participants or []),
    )
    return MBART_SYSTEM_PROMPT.format(
        source_lang=ctx.source_lang,
        meeting_title=ctx.meeting_title,
        participants=ctx.participants_str(),
    ).strip()


@contextmanager
def meeting_context(
    *,
    source_lang: str = "auto",
    meeting_title: str = "Untitled meeting",
    participants: list[str] | None = None,
) -> Iterator[MeetingContext]:
    """Bind meeting metadata for the current translate/summarize call."""
    ctx = MeetingContext(
        source_lang=source_lang or "auto",
        meeting_title=meeting_title or "Untitled meeting",
        participants=list(participants or []),
    )
    token = _CTX.set(ctx)
    try:
        yield ctx
    finally:
        _CTX.reset(token)


def current_context() -> MeetingContext | None:
    return _CTX.get()


def _protect_spans(text: str) -> tuple[str, dict[str, str]]:
    """Stash glossary placeholders and bare numbers so cleanup cannot alter them."""
    mapping: dict[str, str] = {}
    out = text

    def _stash(match: re.Match[str], prefix: str) -> str:
        key = f"⟦{prefix}{len(mapping)}⟧"
        mapping[key] = match.group(0)
        return key

    out = _GLOSSARY_PH_RE.sub(lambda m: _stash(m, "G"), out)
    # Standalone numbers / dates (keep exact).
    out = re.sub(
        r"\b\d+(?:[.,/]\d+)*\b",
        lambda m: _stash(m, "N"),
        out,
    )
    return out, mapping


def _restore_spans(text: str, mapping: dict[str, str]) -> str:
    out = text
    for key, val in mapping.items():
        out = out.replace(key, val)
    return out


def _apply_taglish_map(text: str) -> str:
    out = text
    for pattern, repl in _TAGLISH_MAP:
        out = pattern.sub(repl, out)
    return out


def _label_decision_action(text: str) -> str:
    """Prefix clear decision/action clauses without inventing new facts."""
    raw = (text or "").strip()
    if not raw:
        return raw
    # Already labeled.
    if re.match(r"^(?:NAPAGPASYAHAN|AKSYON)\s*:", raw, re.I):
        return raw

    if _ACTION_CUE_RE.search(raw) and (
        _PERSON_RE.search(raw) or re.search(r"\baksyon\b", raw, re.I)
    ):
        person = ""
        m = _PERSON_RE.search(raw)
        if m:
            person = m.group(1).strip()
        deadline = ""
        d = _DEADLINE_RE.search(raw)
        if d:
            deadline = d.group(1).strip()
        task = re.sub(
            r"^(?:sige[,.]?\s*)?(?:i-?record natin na\s+|irekord natin na\s+)?",
            "",
            raw,
            flags=re.I,
        )
        task = re.sub(
            r"^(?:ang\s+)?(?:action item|aksyon)\s+ay\s+",
            "",
            task,
            flags=re.I,
        )
        task = re.sub(r"\s{2,}", " ", task).strip(" –-.,;") or raw
        # Avoid duplicating the person when already present in the clause.
        if person and person.lower() in task.lower():
            return f"AKSYON: {task}"
        parts = [p for p in (person, task) if p]
        if deadline and deadline.lower() not in task.lower():
            parts.append(deadline)
        return "AKSYON: " + " – ".join(parts)

    if _DECISION_CUE_RE.search(raw):
        body = re.sub(
            r"^(?:napagpasyahan|napagkasunduan)\s*(?:na|:)?\s*",
            "",
            raw,
            flags=re.I,
        ).strip()
        return f"NAPAGPASYAHAN: {body or raw}"

    return raw


def normalize_for_mbart(
    text: str,
    *,
    source_lang: str | None = None,
    apply_taglish: bool = True,
    label_minutes: bool = True,
) -> str:
    """Apply MBART_SYSTEM_PROMPT rules as deterministic pre-mBART cleanup."""
    raw = (text or "").strip()
    if not raw:
        return ""

    ctx = current_context()
    lang = (source_lang or (ctx.source_lang if ctx else "auto") or "auto").strip().lower()

    protected, mapping = _protect_spans(raw)
    out = protected
    out = _FILLER_RE.sub(" ", out)
    # Collapse false starts a few times.
    for _ in range(3):
        nxt = _FALSE_START_RE.sub(r"\1", out)
        if nxt == out:
            break
        out = nxt
    out = _ELLIPSIS_RE.sub("... ", out)
    out = _WS_RE.sub(" ", out).strip(" ,;")

    if apply_taglish and lang in {
        "tl",
        "tagalog",
        "fil",
        "filipino",
        "hil",
        "hiligaynon",
        "ilonggo",
        "auto",
        "tl_xx",
        "id",
        "id_id",
    }:
        out = _apply_taglish_map(out)

    out = _restore_spans(out, mapping)
    out = _WS_RE.sub(" ", out).strip()

    if label_minutes:
        # Label per sentence-ish clause so we do not merge unrelated ideas.
        pieces = re.split(r"(?<=[.!?])\s+", out)
        labeled = [_label_decision_action(p) for p in pieces if p.strip()]
        out = " ".join(labeled).strip() or out

    if out != raw:
        logger.debug(
            "mBART dialect normalize lang=%s title=%s: %r → %r",
            lang,
            ctx.meeting_title if ctx else None,
            raw[:80],
            out[:80],
        )
    return out


def prompt_for_current_context() -> str:
    ctx = current_context() or MeetingContext()
    return render_system_prompt(
        source_lang=ctx.source_lang,
        meeting_title=ctx.meeting_title,
        participants=ctx.participants,
    )
