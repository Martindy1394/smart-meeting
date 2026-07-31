"""Do-not-translate glossary: protect proper nouns across NLLB/mBART."""
from __future__ import annotations

import json
import re
from typing import Any, Iterable


_PLACEHOLDER_TMPL = "⟦SMG{n}⟧"


def load_glossary(raw: Any) -> list[str]:
    """Parse glossary from JSON list, newline/comma text, or list."""
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            items = parsed if isinstance(parsed, list) else [text]
        except (json.JSONDecodeError, TypeError):
            if "\n" in text:
                items = text.splitlines()
            else:
                items = [p.strip() for p in text.split(",")]
    else:
        return []
    seen: set[str] = set()
    out: list[str] = []
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
    # Longest first so "San Carlos City" wins over "Carlos".
    out.sort(key=len, reverse=True)
    return out


def dump_glossary(terms: Iterable[Any] | None) -> str:
    return json.dumps(load_glossary(list(terms or [])), ensure_ascii=False)


def protect(text: str, glossary: list[str]) -> tuple[str, dict[str, str]]:
    """Replace glossary terms with placeholders before MT."""
    if not text or not glossary:
        return text or "", {}
    mapping: dict[str, str] = {}
    out = text
    for i, term in enumerate(glossary):
        if not term or term not in out and term.casefold() not in out.casefold():
            # Case-insensitive search.
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            if not pattern.search(out):
                continue
        placeholder = _PLACEHOLDER_TMPL.format(n=i)
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        out, n = pattern.subn(placeholder, out, count=0)
        if n:
            mapping[placeholder] = term
    return out, mapping


def restore(text: str, mapping: dict[str, str]) -> str:
    """Restore original glossary spellings after MT."""
    if not text or not mapping:
        return text or ""
    out = text
    for placeholder, term in mapping.items():
        out = out.replace(placeholder, term)
        # Models sometimes alter brackets / spacing.
        loose = re.sub(r"\s+", "", placeholder)
        out = out.replace(loose, term)
    return out
