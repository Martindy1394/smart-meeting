"""MT pre/post-processing and NLLB/mBART source-tag helpers (no model weights)."""
from __future__ import annotations


def test_taglish_i_prefix_not_split_as_english_pronoun():
    from app.services.llm import _normalize_spoken_transcript

    src = (
        "Pwede ba nating i-move ang botohan sa susunod na linggo "
        "dahil kulang pa ang dokumento?"
    )
    norm = _normalize_spoken_transcript(src)
    assert "i-move" in norm.lower()
    # Must stay one clause around the Taglish verb — not split before/after it.
    assert "nating. i-move" not in norm.lower()
    assert "nating. I-move" not in norm
    assert "i-move. ang" not in norm.lower()
    assert "i-move ang" in norm.lower()


def test_because_discourse_marker_kept():
    from app.services.llm import _normalize_spoken_transcript

    src = "We need this because of the report and because the committee is late."
    norm = _normalize_spoken_transcript(src)
    assert "because" in norm.lower()
    assert "the report" in norm.lower()
    assert "committee" in norm.lower()


def test_normalize_mt_english_does_not_strip_because():
    from app.services.llm import _normalize_mt_english

    src = "We approved it because the committee finished the report."
    assert _normalize_mt_english(src) == src


def test_nllb_src_code_routes_hiligaynon_auto():
    from app.services.llm import _nllb_src_code

    assert _nllb_src_code("hil") == "ceb_Latn"
    assert _nllb_src_code("tl") == "tgl_Latn"
    hil_text = "Wala gid kami sang budget subong. Indi amo ina."
    assert _nllb_src_code("auto", hil_text) == "ceb_Latn"
    tl_text = "Magandang umaga sa lahat. Kailangan nating aprubahan ang budget."
    assert _nllb_src_code("auto", tl_text) == "tgl_Latn"


def test_hiligaynon_markers_visible_to_language_scores():
    from app.services.llm import _language_scores

    en, fi = _language_scores("Indi gid kita makadto sang meeting subong.")
    assert fi > 0.0


def test_codeswitch_and_perhaps_split():
    from app.services.llm import _normalize_spoken_transcript

    src = (
        "pag-ibig na walang hangganan isang tao lang ako and perhaps "
        "we have to admit our mistakes"
    )
    norm = _normalize_spoken_transcript(src)
    assert "ako. " in norm or "ako." in norm
    assert "perhaps" in norm.lower()


def test_garbled_ph_asr_lyric_salad_detected():
    from app.services.llm import _source_looks_like_garbled_ph_asr

    salad = (
        "pag-ibig na walang andanan itulay na aking nararang tamanan "
        "pagkatikaw na ang aking mahanan tanay kon kaibigan al para sa lahat "
        "itulangan nating itayo isang tao lang ako"
    )
    assert _source_looks_like_garbled_ph_asr(salad) is True
    clean = "Kailangan nating aprubahan ang budget para sa susunod na linggo."
    assert _source_looks_like_garbled_ph_asr(clean) is False


def test_garbled_lyric_kept_untranslated_not_garden_hallucination():
    """ACCO-style sung ASR must not become fluent English nonsense."""
    from app.services import llm

    src = (
        "pag-ibig na walang andanan itulay na aking nararang tamanan "
        "pagkatikaw na ang aking mahanan tanay kon kaibigan al para sa lahat "
        "itulangan nating itayo isang tao lang ako and perhaps we have to of "
        "course admit our mistakes and somehow eventually we will be without "
        "our actions.... don't need to pretend.... don't need to assume.... "
        "everything.... all we have to do is to believe you.... and for us to "
        "be able to do that we have to somehow"
    )
    result = llm.translate(src, target_language="en", source_language="auto")
    text = (result.text or "").lower()
    assert "garden" not in text
    assert "[untranslated:" in (result.text or "").lower() or "untranslated" in (
        result.engine or ""
    ).lower() or any(
        "untranslated" in (r.get("section") or "").lower()
        for r in (result.review_lines or [])
    )
    # English tail should still pass through.
    assert "don't need to pretend" in text or "admit our mistakes" in text
    # Short clear EN lines should not flood Language review.
    en_review = [
        r
        for r in (result.review_lines or [])
        if (r.get("section") == "Language review")
        and "don't need to pretend" in (r.get("line") or "").lower()
    ]
    assert en_review == []
