"""Google Cloud Translation for Hiligaynon → English.

NLLB / mBART do not cover Hiligaynon. Google added ``hil`` as a supported
language mid-2024; this module is the preferred Hiligaynon MT path when
credentials are configured.

Auth (any one):
  * ``GOOGLE_APPLICATION_CREDENTIALS`` pointing at a service-account JSON
  * Application Default Credentials on GCP
  * ``GOOGLE_TRANSLATE_API_KEY`` for the v2 REST API (dev / smaller deploys)

Set ``GOOGLE_TRANSLATE_ENABLED=false`` to force the NLLB ceb_Latn fallback.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache

from ..config import settings

logger = logging.getLogger("smart_meeting.google_translate")

# Google Cloud Translation language code for Hiligaynon (ISO 639-2/3: hil).
_HIL_CODE = "hil"
_EN_CODE = "en"


class GoogleTranslateUnavailable(RuntimeError):
    """Raised when Google Translate cannot be used (disabled / no creds / error)."""


def is_configured() -> bool:
    """True when Hiligaynon→Google routing is enabled and credentials look present."""
    if not bool(getattr(settings, "google_translate_enabled", True)):
        return False
    if (os.environ.get("GOOGLE_TRANSLATE_API_KEY") or "").strip():
        return True
    if (os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or "").strip():
        return True
    # ADC may still work on GCE/Cloud Run without an env path.
    if (getattr(settings, "google_cloud_project", "") or "").strip():
        return True
    return False


@lru_cache(maxsize=1)
def _v3_client():
    from google.cloud import translate_v3 as translate  # type: ignore

    return translate.TranslationServiceClient()


def translate_hiligaynon_to_english(text: str) -> str:
    """Translate Hiligaynon text to English via Google Cloud Translation.

    Raises ``GoogleTranslateUnavailable`` when disabled, unconfigured, or the
    API call fails — callers should fall back to NLLB ``ceb_Latn``.
    """
    raw = (text or "").strip()
    if not raw:
        return ""
    if not bool(getattr(settings, "google_translate_enabled", True)):
        raise GoogleTranslateUnavailable("Google Translate disabled in settings")

    api_key = (os.environ.get("GOOGLE_TRANSLATE_API_KEY") or "").strip()
    if api_key:
        return _translate_v2_api_key(raw, api_key)

    project = (getattr(settings, "google_cloud_project", "") or "").strip() or (
        os.environ.get("GOOGLE_CLOUD_PROJECT") or ""
    ).strip()
    if not project and not (os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or "").strip():
        raise GoogleTranslateUnavailable(
            "Set GOOGLE_CLOUD_PROJECT / GOOGLE_APPLICATION_CREDENTIALS "
            "or GOOGLE_TRANSLATE_API_KEY for Hiligaynon translation"
        )

    try:
        return _translate_v3(raw, project)
    except GoogleTranslateUnavailable:
        raise
    except Exception as exc:
        logger.warning("Google Translate v3 failed (%s); Hiligaynon MT unavailable", exc)
        raise GoogleTranslateUnavailable(str(exc)) from exc


def _translate_v3(text: str, project: str) -> str:
    try:
        from google.cloud import translate_v3 as translate  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise GoogleTranslateUnavailable(
            "google-cloud-translate not installed. "
            "pip install google-cloud-translate"
        ) from exc

    if not project:
        # Parent is required for v3; without a project try location-less via ADC default.
        raise GoogleTranslateUnavailable("GOOGLE_CLOUD_PROJECT is required for Translation API v3")

    client = _v3_client()
    parent = f"projects/{project}/locations/global"
    response = client.translate_text(
        request={
            "parent": parent,
            "contents": [text],
            "mime_type": "text/plain",
            "source_language_code": _HIL_CODE,
            "target_language_code": _EN_CODE,
        }
    )
    translations = list(response.translations or [])
    if not translations:
        raise GoogleTranslateUnavailable("Google Translate returned no translations")
    out = (translations[0].translated_text or "").strip()
    if not out:
        raise GoogleTranslateUnavailable("Google Translate returned empty text")
    return out


def _translate_v2_api_key(text: str, api_key: str) -> str:
    """Lightweight v2 REST path for API-key auth (no ADC)."""
    import json
    import urllib.error
    import urllib.parse
    import urllib.request

    params = urllib.parse.urlencode(
        {
            "q": text,
            "source": _HIL_CODE,
            "target": _EN_CODE,
            "format": "text",
            "key": api_key,
        }
    )
    url = f"https://translation.googleapis.com/language/translate/v2?{params}"
    req = urllib.request.Request(url, method="POST", data=b"")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:300]
        raise GoogleTranslateUnavailable(f"Google Translate v2 HTTP {exc.code}: {body}") from exc
    except Exception as exc:
        raise GoogleTranslateUnavailable(str(exc)) from exc

    try:
        out = payload["data"]["translations"][0]["translatedText"]
    except (KeyError, IndexError, TypeError) as exc:
        raise GoogleTranslateUnavailable(f"Unexpected v2 response: {payload!r}") from exc
    out = (out or "").strip()
    if not out:
        raise GoogleTranslateUnavailable("Google Translate v2 returned empty text")
    return out
