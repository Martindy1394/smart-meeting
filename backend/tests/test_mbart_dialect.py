"""Pre-mBART dialect normalizer (SmartScribe MBART_SYSTEM_PROMPT rules)."""


def test_system_prompt_renders_meeting_context():
    from app.services.mbart_dialect import render_system_prompt

    prompt = render_system_prompt(
        source_lang="hil",
        meeting_title="Board Budget Review",
        participants=["Maria", "Juan"],
    )
    assert "Hiligaynon or Filipino" in prompt or "dialect normalizer" in prompt
    assert "Board Budget Review" in prompt
    assert "Maria" in prompt and "Juan" in prompt
    assert "Input language: hil" in prompt


def test_removes_fillers_and_false_starts():
    from app.services.mbart_dialect import normalize_for_mbart

    raw = "Uh, um, kailangan natin ang ang budget, 'di ba, kuan."
    out = normalize_for_mbart(raw, source_lang="tl", label_minutes=False)
    assert "uh" not in out.lower()
    assert "kuan" not in out.lower()
    assert "ang ang" not in out.lower()
    assert "budget" in out.lower()


def test_preserves_glossary_placeholders_and_numbers():
    from app.services.mbart_dialect import normalize_for_mbart

    raw = "Si ⟦SMG0⟧ ang responsable sa 3 action item bago ang 2026."
    out = normalize_for_mbart(raw, source_lang="tl")
    assert "⟦SMG0⟧" in out
    assert "3" in out
    assert "2026" in out


def test_taglish_stabilized_for_mbart():
    from app.services.mbart_dialect import normalize_for_mbart

    raw = "Pwede ba nating i-move ang botohan sa susunod na linggo?"
    out = normalize_for_mbart(raw, source_lang="tl", label_minutes=False)
    assert "i-move" not in out.lower()
    assert "ilipat" in out.lower()


def test_decision_and_action_labels():
    from app.services.mbart_dialect import normalize_for_mbart

    decision = "Napagpasyahan na aprubahan ang budget."
    out_d = normalize_for_mbart(decision, source_lang="tl")
    assert out_d.startswith("NAPAGPASYAHAN:")

    action = "Ang action item ay kay Juan bago ang Biyernes."
    out_a = normalize_for_mbart(action, source_lang="tl")
    assert out_a.startswith("AKSYON:")
    assert "Juan" in out_a


def test_meeting_context_binds_for_prompt():
    from app.services import mbart_dialect

    with mbart_dialect.meeting_context(
        source_lang="tl",
        meeting_title="ACCO",
        participants=["Ana"],
    ):
        prompt = mbart_dialect.prompt_for_current_context()
        assert "ACCO" in prompt
        assert "Ana" in prompt
        assert "Input language: tl" in prompt


def test_greeting_not_force_labeled():
    from app.services.mbart_dialect import normalize_for_mbart

    out = normalize_for_mbart(
        "Magandang umaga sa lahat ng dumalo sa meeting.",
        source_lang="tl",
    )
    assert not out.startswith("NAPAGPASYAHAN:")
    assert not out.startswith("AKSYON:")
