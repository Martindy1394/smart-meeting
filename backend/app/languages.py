"""Supported translation languages and helpers.

``mbart_code`` maps short codes to ``facebook/mbart-large-50-many-to-many-mmt``
tokens for non-PH targets.

Philippine → English uses **NLLB** (``tgl_Latn`` Tagalog / ``ceb_Latn`` for
Hiligaynon-closest Visayan) because mBART has no reliable Tagalog source —
``tl_XX`` often yields Pashto, and ``id_ID`` invents unrelated English.
"""
from __future__ import annotations

# code -> (display name, mBART-50 language token, fallback flag)
#
# NOTE: Philippine→English is handled by NLLB in ``llm.py``. mBART ``id_ID``
# remains only as a last-resort fallback for PH text.
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
    # Hiligaynon / Tagalog: NLLB primary; mBART id_ID is fallback only.
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
