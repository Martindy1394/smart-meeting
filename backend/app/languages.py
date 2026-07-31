"""Supported translation languages and helpers.

``mbart_code`` maps short codes to ``facebook/mbart-large-50-many-to-many-mmt``
language tokens.

Tagalog (``tl``)
  Stock mBART-50 **includes** native ``tl_XX``. An older inherited workaround
  tagged Tagalog as Indonesian ``id_ID`` whenever no PH fine-tune was loaded.
  Live fixture benchmark (``scripts/ph_mt/benchmark_mbart_tags.py``, 2026-07-31)
  measured higher token-F1 for ``tl_XX`` (0.43) than ``id_ID`` (0.34), so stock
  Tagalog now uses ``tl_XX``. Optional ``MBART_PH_FINE_TUNED_MODEL`` checkpoints
  are trained on ``tl_XX`` → ``en_XX`` (see ``scripts/ph_mt/finetune_mbart.py``).

Hiligaynon (``hil``)
  Stock mBART-50 has **no** ``hil_XX``. Fine-tunes trained with ``--lang hil``
  add ``hil_XX`` (embedding init from ``tl_XX``) and set
  ``finetune_meta.json: has_hil_xx=true``; ``mbart_code`` then returns
  ``hil_XX``. Without that flag, Hiligaynon still degrades to ``id_ID`` (stock)
  or legacy ``tl_XX`` proxy (older PH fine-tunes). Production Hiligaynon →
  English must still prefer Google Cloud Translation first.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

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


def ensure_tokenizer_hil_xx(tokenizer) -> bool:
    """Re-bind ``hil_XX`` into ``lang_code_to_id`` after loading a PH fine-tune.

    mBART-50's built-in map omits ``hil_XX``; fine-tunes add it as an extra
    special token, but ``MBart50TokenizerFast`` does not restore the lang-code
    entry on ``from_pretrained``. Returns True when ``hil_XX`` is available.
    """
    try:
        hil_id = tokenizer.convert_tokens_to_ids("hil_XX")
    except Exception:
        return False
    unk = getattr(tokenizer, "unk_token_id", None)
    if hil_id is None or hil_id == unk:
        return False
    lang_map = getattr(tokenizer, "lang_code_to_id", None)
    if not isinstance(lang_map, dict):
        return False
    lang_map["hil_XX"] = hil_id
    if hasattr(tokenizer, "id_to_lang_code"):
        tokenizer.id_to_lang_code[hil_id] = "hil_XX"
    return True


@lru_cache(maxsize=8)
def ph_finetune_has_hil_xx(model_path: str = "") -> bool:
    """True when ``MBART_PH_FINE_TUNED_MODEL`` advertises a real ``hil_XX`` token."""
    from .config import settings

    path = (model_path or settings.mbart_ph_finetuned_model or "").strip()
    if not path:
        return False
    meta = Path(path) / "finetune_meta.json"
    if not meta.is_file():
        return False
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(data.get("has_hil_xx"))


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

    ph_ft = (settings.mbart_ph_finetuned_model or "").strip()
    if ph_ft and normalized == "tl":
        return "tl_XX"
    if ph_ft and normalized == "hil":
        # Real vocab slot when trained with --lang hil; else legacy tl_XX proxy.
        return "hil_XX" if ph_finetune_has_hil_xx(ph_ft) else "tl_XX"
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
