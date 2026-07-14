"""Supported translation languages and helpers.

The application supports (at least) the 14 languages required by the spec.
``mbart_code`` maps our short codes to the token identifiers expected by
``facebook/mbart-large-50-many-to-many-mmt``.  Languages without a native
mBART code (Hiligaynon) fall back to the closest available model language while
still being offered in the UI.
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
    # Hiligaynon has no dedicated mBART-50 token; Tagalog (tl_XX) is the closest
    # supported Philippine language and is used as the fallback for the model.
    "hil": {"name": "Hiligaynon", "mbart": "tl_XX", "fallback": True},
    "tl": {"name": "Tagalog", "mbart": "tl_XX"},
    # English is always available as a convenience target.
    "en": {"name": "English", "mbart": "en_XX"},
}


def language_name(code: str) -> str:
    entry = LANGUAGES.get(code)
    if entry:
        return entry["name"]
    return code


def mbart_code(code: str) -> str | None:
    entry = LANGUAGES.get(code)
    if not entry:
        return None
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
