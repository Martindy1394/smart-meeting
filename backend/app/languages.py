"""Supported translation languages and helpers.

``mbart_code`` maps short codes to ``facebook/mbart-large-50-many-to-many-mmt``
language tokens.

Tagalog (``tl``)
  Stock mBART-50 **includes** native ``tl_XX``. An older inherited workaround
  tagged Tagalog as Indonesian ``id_ID`` whenever no PH fine-tune was loaded.
  Live fixture benchmark (``scripts/ph_mt/benchmark_mbart_tags.py``, 2026-07-31)
  measured higher token-F1 for ``tl_XX`` (0.43) than ``id_ID`` (0.34), so stock
  Tagalog now uses ``tl_XX``. Optional ``MBART_PH_FINE_TUNED_MODEL`` checkpoints
  are also trained on ``tl_XX`` → ``en_XX``.

Hiligaynon (``hil``)
  mBART-50 has **no** ``hil_XX`` (or any Hiligaynon) token — confirmed against
  ``tokenizer.lang_code_to_id``. Proxying as ``id_ID`` / ``tl_XX`` is a
  **degraded last resort** only. Production Hiligaynon → English must go through
  Google Cloud Translation (``hil``) first, then NLLB ``ceb_Latn``, then mBART.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("smart_meeting.languages")

# Shorthand aliases accepted by routers / ASR language fields.
_CODE_ALIASES: dict[str, str] = {
    "fil": "tl",
    "filipino": "tl",
    "tagalog": "tl",
    "hiligaynon": "hil",
    "ilonggo": "hil",
}

# code -> (display name, mBART-50 language token, fallback flag)
LANGUAGES: dict[str, dict] = {
    "es": {"name": "Spanish", "mbart": "es_XX"},
    "fr": {"name": "French", "mbart": "fr_XX"},
    "de": {"name": "German", "mbart": "de_DE"},
    "it": {"name": "Italian", "mbart": "it_IT"},
    "pt": {"name": "Portuguese", "mbart": "pt_XX"},
    "ar": {"name": "Arabic", "mbart": "ar_AR"},
    "hi": {"name": "Hindi", "mbart": "hi_IN"},
    "ja": {"name": "Japanese", "mbart": "ja_XX"},
    "zh": {"name": "Chinese", "mbart": "zh_CN"},
    "ru": {"name": "Russian", "mbart": "ru_RU"},
    "nl": {"name": "Dutch", "mbart": "nl_XX"},
    "ko": {"name": "Korean", "mbart": "ko_KR"},
    "id": {"name": "Indonesian", "mbart": "id_ID"},
    # Native mBART-50 Tagalog token (preferred over historical id_ID workaround).
    "tl": {"name": "Tagalog", "mbart": "tl_XX"},
    # No hil_XX in mBART-50 — id_ID is a degraded typological proxy only.
    "hil": {"name": "Hiligaynon", "mbart": "id_ID", "fallback": True},
    # English is always available as a convenience target.
    "en": {"name": "English", "mbart": "en_XX"},
}

# Codes the MT stack must always resolve (startup + tests).
REQUIRED_MBART_SHORT_CODES: tuple[str, ...] = ("en", "tl", "id", "hil")


def normalize_lang_code(code: str | None) -> str:
    """Lowercase + alias-normalize a language shorthand."""
    raw = (code or "").strip().lower()
    if not raw:
        return ""
    return _CODE_ALIASES.get(raw, raw)


def language_name(code: str) -> str:
    entry = LANGUAGES.get(normalize_lang_code(code))
    if entry:
        return entry["name"]
    return code


def mbart_code(code: str) -> str | None:
    """Map a short language code to an mBART-50 ``lang_code_to_id`` key.

    Returns ``None`` for unrecognized codes (callers must not silently treat
    that as English without logging — see ``_mbart_translate``).
    """
    from .config import settings

    normalized = normalize_lang_code(code)
    entry = LANGUAGES.get(normalized)
    if not entry:
        return None

    # Fine-tuned PH mBART was trained with tl_XX for Tagalog + Hiligaynon proxy.
    # For Hiligaynon that still is not a real vocabulary slot — it only helps
    # when the checkpoint saw Hiligaynon text under the tl_XX tag during LoRA.
    if (settings.mbart_ph_finetuned_model or "").strip() and normalized in {
        "tl",
        "hil",
    }:
        return "tl_XX"
    return entry.get("mbart")


def assert_mbart_codes_resolvable() -> None:
    """Startup guard: every shorthand the MT router passes must resolve."""
    missing = [c for c in REQUIRED_MBART_SHORT_CODES if not mbart_code(c)]
    if missing:
        raise RuntimeError(
            f"mBART language map missing required codes: {missing}. "
            "Check app/languages.py LANGUAGES."
        )
    # Aliases used by ASR / meeting.language fields.
    for alias in ("fil", "tagalog", "hiligaynon", "ilonggo"):
        if not mbart_code(alias):
            raise RuntimeError(f"mBART language alias {alias!r} does not resolve")


def list_languages() -> list[dict]:
    return [
        {
            "code": code,
            "name": entry["name"],
            "fallback": entry.get("fallback", False),
        }
        for code, entry in LANGUAGES.items()
    ]
