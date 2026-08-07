"""Exact Tagalog meeting phrase lexicon (no model weights)."""


def test_lexicon_loaded_and_exact_match():
    from app.services import tagalog_phrases

    assert tagalog_phrases.lexicon_size() >= 50
    src = "Magandang umaga sa lahat ng dumalo sa meeting."
    assert (
        tagalog_phrases.lookup_exact(src)
        == "Good morning to everyone who attended the meeting."
    )


def test_pipeline_uses_phrase_lexicon_for_seed_line():
    from app.services import llm

    src = "Kailangan nating aprubahan ang budget para sa susunod na quarter."
    tr = llm.translate(src, target_language="en", source_language="tl")
    assert tr.engine == "tagalog-phrase-lexicon"
    assert "budget" in (tr.text or "").lower()
    assert "quarter" in (tr.text or "").lower()


def test_unknown_tagalog_not_forced_to_lexicon():
    from app.services import tagalog_phrases

    assert tagalog_phrases.lookup_exact("Ito ay isang bagong pangungusap na wala sa seed.") is None
