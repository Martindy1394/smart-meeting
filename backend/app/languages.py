"""Supported translation languages and helpers.

The application supports (at least) the 14 languages required by the spec.
``mbart_code`` maps our short codes to the token identifiers expected by
``facebook/mbart-large-50-many-to-many-mmt``.

Hiligaynon / Tagalog do not have a reliable mBART source token on this
checkpoint (``tl_XX`` often yields Pashto instead of English), so they map to
Indonesian ``id_ID`` — the closest working Austronesian language in the model.
"""
from __future__ import annotations

# code -> (display name, mBART-50 language token, fallback flag)
#
# NOTE: ``tl_XX`` exists in the tokenizer vocabulary but this mBART-50 checkpoint
# frequently ignores ``forced_bos_token_id=en_XX`` when the source is ``tl_XX``
# (or identity ``en_XX→en_XX``) and emits Pashto/Telugu/Hindi instead. For
# Philippine speech we therefore use Indonesian ``id_ID`` — a related
# Austronesian language that reliably decodes to English.
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
    # Hiligaynon / Tagalog: no reliable mBART source token — use Indonesian.
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
