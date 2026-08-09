"""mBART-only Hiligaynon mistag / ASR-as-Tagalog helpers."""


def test_hiligaynon_cue_detection():
    from app.services.mbart_hiligaynon import looks_hiligaynon_heavy

    assert looks_hiligaynon_heavy(
        "Indi gid naton magdesisyon subong kon wala pa ang report."
    )
    assert looks_hiligaynon_heavy(
        "Hindi gid natin magdesisyon ngayon kung wala pa ang report ng komite."
    )
    assert not looks_hiligaynon_heavy(
        "We need to approve the budget for the next quarter."
    )


def test_cognate_bridge_for_tl_encode():
    from app.services.mbart_hiligaynon import stabilize_hiligaynon_for_tl_mbart

    src = "Maayong aga sa tanan nga nag-atendir sang miting."
    out = stabilize_hiligaynon_for_tl_mbart(src).lower()
    assert "magandang umaga" in out
    assert "lahat" in out
    assert "maayong" not in out


def test_hiligaynon_lexicon_covers_fixtures():
    from app.services import hiligaynon_phrases

    assert hiligaynon_phrases.lexicon_size() >= 7
    hit = hiligaynon_phrases.lookup_exact(
        "Maayong aga sa tanan nga nag-atendir sang miting."
    )
    assert hit and "morning" in hit.lower()


def test_mbart_path_uses_hiligaynon_lexicon_before_model():
    from unittest.mock import patch

    from app.services import llm

    src = "Kinahanglan naton aprubahan ang budget para sa masunod nga quarter."

    def _boom(*a, **k):
        raise AssertionError("mBART must not load on Hiligaynon lexicon hit")

    with patch.object(llm, "_mbart_generate_text", side_effect=_boom):
        out = llm._mbart_translate(src, "hil", "en")
    assert "budget" in out.lower()
    assert "quarter" in out.lower()


def test_noise_fixture_lexicon_hit_under_tl_tag():
    """ASR-Tagalogized Hiligaynon still short-circuits inside mBART path."""
    from unittest.mock import patch

    from app.services import llm

    src = "May tanong ba parte sa agenda?"

    def _boom(*a, **k):
        raise AssertionError("model must not load on noise-fixture lexicon hit")

    with patch.object(llm, "_mbart_generate_text", side_effect=_boom):
        out = llm._mbart_translate(src, "tl", "en")
    assert "question" in out.lower() or "agenda" in out.lower()
