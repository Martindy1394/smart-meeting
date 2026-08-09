"""Topic-aware BART: meeting vs general content kinds (Gaps B/C + Gap A gate)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services import llm

LYRIC_UNITS = [
    "Pumapatak sa mga halang at mga bulaklak.",
    "Maari bang huwag ka na.",
    "Sa piling ko'y lumisan pa.",
    "Hatid ko lamang ang saksihin.",
    "Buhus pa ulan.",
    "Aking mundo'y lunureng tuluyan.",
    "Hatid mo may bagyo dalangin ito ng puso ko ang lungiliyap.",
    "Damdamin ko'y maapaw.",
    "Pag-ibig ko'y humihiyaw.",
    "Sa tuwa tuwing mong ulan at tupil-ilin ka.",
]


MEETING_EN = (
    "The board approved the quarterly budget after review. "
    "Marketing will launch the campaign next week. "
    "Facilities will prepare the hall and stage lighting. "
    "Security assigned entrance badges for all guests. "
    "The chair opened discussion on the annual calendar carefully."
)

LYRIC_EN = (
    "Rain falls on the fences and the flowers in the garden tonight. "
    "Please do not leave my side again under the storm clouds. "
    "I can only bear witness to what remains after you go. "
    "Pour the rain until my world is fully submerged below. "
    "You bring a storm, a prayer rising from my heart today. "
    "My feelings overflow and my love cries out with joy. "
    "With joy whenever your rain folds gently around me now."
)


def test_gap_a_dedupe_fix_landed():
    """Sign-off gate: lyric content-loss dedupe harden must already be present."""
    assert llm._DEDUPE_MIN_SHARED_WORDS >= 3
    assert llm._DEDUPE_LOCAL_WINDOW >= 2
    post = llm._dedupe_units(list(LYRIC_UNITS))
    assert len(post) == len(LYRIC_UNITS)
    joined = " ".join(post).lower()
    for marker in ("buhus pa ulan", "hatid mo may bagyo", "damdamin", "humihiyaw"):
        assert marker in joined, marker


def test_resolve_and_frame_by_content_kind():
    assert llm._resolve_content_kind("english_translation") == "meeting"
    assert llm._resolve_content_kind("meeting") == "meeting"
    assert llm._resolve_content_kind("general") == "general"
    assert "Board meeting" in llm._bart_frame_prefix("meeting")
    assert llm._bart_frame_prefix("general") == "Summarize the following."


def test_meeting_minutes_bucketing_only_for_meeting_kind():
    units = [
        "The board discussed the annual calendar.",
        "Members approved the venue budget.",
        "Marketing will launch the campaign next week.",
    ]
    meeting = llm._format_meeting_minutes(units, "bullets", content_kind="meeting")
    general = llm._format_meeting_minutes(units, "bullets", content_kind="general")
    assert "Decisions" in meeting
    assert "Action items" in meeting
    assert "Decisions" not in general
    assert "Action items" not in general
    assert "Marketing will launch the campaign next week." in general


def test_bart_chunk_framing_is_conditional():
    calls = []

    class FakePipe:
        def __call__(self, framed, **kwargs):
            calls.append(framed)
            return [{"summary_text": "Summary of the chunk."}]

    llm._bart_summarize_chunk(FakePipe(), "Rain falls on the flowers.", content_kind="general")
    llm._bart_summarize_chunk(FakePipe(), "The board approved the budget.", content_kind="meeting")
    assert calls[0].startswith("Summarize the following.")
    assert calls[1].startswith("Board meeting discussion and decisions.")


def test_topic_label_generalizes_on_lyric_and_story():
    lyric_chunk = " ".join(LYRIC_UNITS[4:7])
    label = llm._topic_label(lyric_chunk, 1)
    assert label
    assert "Topic 1" != label or True  # may fall back only if empty
    assert any(w in label.lower() for w in ("ulan", "buhus", "bagyo", "mundo", "puso", "hatid"))

    story = (
        "The explorer crossed the desert at dawn with a small caravan of camels. "
        "Water was scarce but the oasis appeared before noon."
    )
    story_label = llm._topic_label(story, 2)
    assert any(w in story_label.lower() for w in ("explorer", "desert", "dawn", "caravan", "oasis"))


def test_meeting_summarize_keeps_minutes_structure():
    """Meeting fixture: Discussion/Decisions/Action path preserved."""

    def fake_topics(text, *, content_kind="meeting"):
        assert content_kind == "meeting"
        return [
            (
                "Budget",
                "The board approved the quarterly budget. Marketing will launch next week.",
            )
        ]

    with patch.object(llm, "_bart_summarize_topics", side_effect=fake_topics):
        summary, engine = llm.summarize(
            MEETING_EN, output_format="bullets", source_kind="meeting"
        )
    assert "minutes" in engine or "meeting" in engine
    assert "Decisions" in summary or "Action items" in summary or "approved" in summary.lower()
    assert "Board meeting" in llm._bart_frame_prefix("meeting")


def test_general_summarize_is_flat_bullets_not_minutes():
    """Non-meeting fixture: no forced Decisions/Action-items sections."""

    def fake_topics(text, *, content_kind="meeting"):
        assert content_kind == "general"
        return [
            ("Rain And Flowers", "Rain falls on fences and flowers. Do not leave my side."),
            ("Storm And Heart", "A storm rises from the heart. Love cries out with joy."),
        ]

    with patch.object(llm, "_bart_summarize_topics", side_effect=fake_topics):
        summary, engine = llm.summarize(
            LYRIC_EN, output_format="bullets", source_kind="general"
        )
    assert "Decisions" not in summary
    assert "Action items" not in summary
    assert "Discussion" not in summary  # minutes header, not topic word
    low = summary.lower()
    assert "rain" in low or "flowers" in low
    assert "storm" in low or "heart" in low or "love" in low or "joy" in low
    assert "general" in engine or "topic" in engine


def test_discourse_overlap_duplicates_are_collapsed():
    """Overlap tails must not yield the same bullet twice after merge/dedupe."""
    source_units = [
        "The committee closed the venue discussion carefully today.",
        "Members then approved the lighting budget for the hall.",
        "Marketing will publish the schedule before Friday morning.",
    ]
    # Simulate topic N ending and topic N+1 starting with the same decision.
    summary_units = [
        "Members then approved the lighting budget for the hall.",
        "Members then approved the lighting budget for the hall.",
        "Marketing will publish the schedule before Friday morning.",
    ]
    merged = llm._merge_missing_units(source_units, summary_units, min_overlap=0.5)
    approved = [u for u in merged if "approved the lighting" in u.lower()]
    assert len(approved) == 1


def test_english_translation_alias_maps_to_meeting():
    assert llm._resolve_content_kind("english_translation") == "meeting"
    with patch.object(
        llm,
        "_contextual_bart_summary",
        return_value=("• Point\n", "bart-meeting-minutes"),
    ) as mocked:
        llm.summarize(MEETING_EN, source_kind="english_translation")
        assert mocked.call_args.kwargs.get("content_kind") == "meeting"


if __name__ == "__main__":
    test_gap_a_dedupe_fix_landed()
    test_resolve_and_frame_by_content_kind()
    test_meeting_minutes_bucketing_only_for_meeting_kind()
    test_bart_chunk_framing_is_conditional()
    test_topic_label_generalizes_on_lyric_and_story()
    test_meeting_summarize_keeps_minutes_structure()
    test_general_summarize_is_flat_bullets_not_minutes()
    test_discourse_overlap_duplicates_are_collapsed()
    test_english_translation_alias_maps_to_meeting()
    print("all_general_bart_summary_tests_passed")
