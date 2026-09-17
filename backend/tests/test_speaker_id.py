"""Introduction → roster speaker identification."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.services import speaker_id  # noqa: E402
from app.services import speaker_memory  # noqa: E402


class SpeakerIdTests(unittest.TestCase):
    def test_extracts_english_and_philippine_intros(self):
        self.assertEqual(speaker_id.extract_introduction("My name is Maria Santos."), "Maria Santos")
        self.assertEqual(speaker_id.extract_introduction("I am Juan Dela Cruz from Iloilo"), "Juan Dela Cruz")
        self.assertEqual(speaker_id.extract_introduction("Ako si Ana Reyes"), "Ana Reyes")
        self.assertEqual(
            speaker_id.extract_introduction("Ang pangalan ko ay Pedro Garcia"),
            "Pedro Garcia",
        )
        self.assertIsNone(speaker_id.extract_introduction("The motion is carried."))
        self.assertIsNone(
            speaker_id.extract_introduction(
                "Hello, coop test. This is cryptocur and I would like to raise my concern"
            )
        )
        self.assertEqual(
            speaker_id.extract_introduction("This is Martin and I would like to raise my concern"),
            "Martin",
        )

    def test_roster_match_and_guest(self):
        roster = ["Maria Santos", "Juan Dela Cruz"]
        name, score, guest = speaker_id.match_roster("Maria Santos", roster)
        self.assertEqual(name, "Maria Santos")
        self.assertGreaterEqual(score, 0.99)
        self.assertFalse(guest)
        name, score, guest = speaker_id.match_roster("Santos", roster)
        self.assertEqual(name, "Maria Santos")
        self.assertFalse(guest)
        name, score, guest = speaker_id.match_roster("Guest Speaker", roster)
        self.assertTrue(guest)
        self.assertEqual(name, "Guest Speaker")

    def test_officer_not_double_counted(self):
        names = speaker_id.roster_names(
            SimpleNamespace(attendees=["Maria Santos", "Ada"], presiding_officer="Maria Santos")
        )
        self.assertEqual(names, ["Maria Santos", "Ada"])

    def test_identify_propagates_across_voice_cluster(self):
        segs = [
            SimpleNamespace(
                text="Ako si Maria Santos.",
                speaker_index=1,
                speaker_label="Voice 1",
                start=0.0,
                end=2.0,
                low_confidence=False,
            ),
            SimpleNamespace(
                text="We will vote tomorrow.",
                speaker_index=1,
                speaker_label="Voice 1",
                start=2.0,
                end=5.0,
                low_confidence=False,
            ),
            SimpleNamespace(
                text="I second the motion.",
                speaker_index=2,
                speaker_label="Voice 2",
                start=5.0,
                end=7.0,
                low_confidence=False,
            ),
        ]
        meeting = SimpleNamespace(
            attendees=["Maria Santos", "Juan Dela Cruz"],
            presiding_officer="Maria Santos",
        )
        speaker_id.identify_segments(segs, meeting)
        self.assertEqual(segs[0].speaker_name, "Maria Santos")
        self.assertEqual(segs[1].speaker_name, "Maria Santos")
        self.assertGreaterEqual(segs[0].speaker_confidence, speaker_id.DISPLAY_THRESHOLD)
        self.assertFalse(hasattr(segs[2], "speaker_name") and segs[2].speaker_name == "Maria Santos")

        report = speaker_id.build_attendance_report(segs, meeting)
        present = {p.name for p in report.present}
        self.assertIn("Maria Santos", present)
        self.assertIn("Juan Dela Cruz", report.absent)

    def test_correction_alias_is_reused(self):
        tmp = Path(tempfile.mkdtemp())
        orig = speaker_memory._dir
        speaker_memory._dir = lambda: tmp
        try:
            speaker_memory.remember_alias("user-1", "marya", "Maria Santos")
            aliases = speaker_memory.load_aliases("user-1")
            self.assertEqual(aliases["marya"], "Maria Santos")
        finally:
            speaker_memory._dir = orig


if __name__ == "__main__":
    unittest.main()
