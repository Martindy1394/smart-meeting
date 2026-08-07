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
