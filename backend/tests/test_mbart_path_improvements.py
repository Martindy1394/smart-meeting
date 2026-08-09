"""mBART-path-only reliability helpers (does not load the neural model)."""


def test_postprocess_strips_dialect_label_echoes():
    from app.services.llm import _postprocess_mbart_english

    assert (
        _postprocess_mbart_english("ACKNOWLEDGEMENTS: Juan owns the action item.")
        == "Juan owns the action item."
    )
    assert (
        _postprocess_mbart_english("AKSYON: Submit the report by Friday.")
        == "Submit the report by Friday."
    )
    assert (
        _postprocess_mbart_english("NAPAGPASYAHAN: The budget is approved.")
        == "The budget is approved."
    )


def test_postprocess_strips_untranslated_si_marker():
    from app.services.llm import _postprocess_mbart_english

    assert (
        _postprocess_mbart_english("Si Maria is responsible for the follow-up.")
        == "Maria is responsible for the follow-up."
    )


def test_en_normalize_skips_labels_and_taglish_force():
    """EN-target mBART prep must not inject NAPAGPASYAHAN/AKSYON or rewrite Taglish."""
    from app.services.mbart_dialect import normalize_for_mbart

    action = "Sige, i-record natin na ang action item ay kay Juan bago ang deadline."
    out = normalize_for_mbart(
        action,
        source_lang="tl",
        apply_taglish=False,
        label_minutes=False,
    )
    assert not out.startswith("AKSYON:")
    assert not out.startswith("NAPAGPASYAHAN:")
    assert "i-record" in out.lower()

    move = "Pwede ba nating i-move ang botohan sa susunod na linggo?"
    out_m = normalize_for_mbart(
        move,
        source_lang="tl",
        apply_taglish=False,
        label_minutes=False,
    )
    assert "i-move" in out_m.lower()


def test_mbart_path_uses_exact_lexicon_before_model():
    """Curated TL meeting lines short-circuit inside `_mbart_translate` (no model load)."""
    from unittest.mock import patch

    from app.services import llm

    src = "Magandang umaga sa lahat ng dumalo sa meeting."
    expected = "Good morning to everyone who attended the meeting."

    def _boom():
        raise AssertionError("mBART model must not load on exact lexicon hit")

    with patch.object(llm._pipelines, "mbart", side_effect=_boom):
        out = llm._mbart_translate(src, "tl", "en")
    assert out == expected


def test_clear_tagalog_reaches_mbart_not_untranslated_marker():
    """Gap fix: clear Tagalog prose must call mBART, not keep [untranslated:]."""
    from unittest.mock import patch

    from app.config import settings
    from app.services import llm

    prose = (
        "Una sa lahat gusto kong pasalamatan ang lahat ng dumalo sa ating "
        "regular na pagpupulong ngayong umaga upang pag-usapan ang mga isyu "
        "tungkol sa pasilidad."
    )
    calls: list[tuple] = []

    def fake_mbart(text, src, tgt):
        calls.append((text, src, tgt))
        return "First of all I want to thank everyone who attended our regular meeting this morning to discuss facility issues."

    with (
        patch.object(settings, "ph_translate_backend", "mbart"),
        patch.object(llm, "_mbart_translate", side_effect=fake_mbart),
        patch("app.services.tagalog_phrases.lookup_exact", return_value=None),
    ):
        tr = llm._translate_to_english(prose, "tl")

    assert calls, "mBART must be invoked for clear Tagalog prose"
    assert "[untranslated:" not in (tr.text or "").lower()
    assert "thank" in (tr.text or "").lower() or "facility" in (tr.text or "").lower()
