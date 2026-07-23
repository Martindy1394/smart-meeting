"""Supported translation languages and helpers.

``mbart_code`` maps short codes to ``facebook/mbart-large-50-many-to-many-mmt``
tokens for non-PH targets.

Philippine → English defaults to **NLLB** (``tgl_Latn`` / ``ceb_Latn``). When a
fine-tuned mBART checkpoint is configured (``MBART_PH_FINE_TUNED_MODEL``),
Tagalog/Hiligaynon use ``tl_XX`` (Hiligaynon has no native mBART token).
"""
from __future__ import annotations

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
    # Stock mBART: id_ID fallback. Fine-tuned PH checkpoint: tl_XX via mbart_code().
    "hil": {"name": "Hiligaynon", "mbart": "id_ID", "fallback": True},
    "tl": {"name": "Tagalog", "mbart": "id_ID", "fallback": True},
    # English is always available as a convenience target.
    "en": {"name": "English", "mbart": "en_XX"},
}


def language_name(code: str) -> str:
    entry = LANGUAGES.get(code)
    if entry:
        return entry["name"]
    return code


def mbart_code(code: str) -> str | None:
    from .config import settings

    entry = LANGUAGES.get(code)
    if not entry:
        return None
    # Fine-tuned PH mBART was trained with tl_XX for Tagalog + Hiligaynon.
    if (settings.mbart_ph_finetuned_model or "").strip() and code in {
        "tl",
        "hil",
        "fil",
        "filipino",
        "tagalog",
        "hiligaynon",
        "ilonggo",
    }:
        return "tl_XX"
    return entry.get("mbart")

def list_languages() -> list[dict]:
    return [
        {
            "code": code,
            "name": entry["name"],
            "fallback": entry.get("fallback", False),
        }
        for code, entry in LANGUAGES.items()
    ]
