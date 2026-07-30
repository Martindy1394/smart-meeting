"""Segment / attendees / AI-quality model alignment tests."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


class AttendeesBridgeTests(unittest.TestCase):
    def test_roundtrip_list_and_json(self):
        from app.services.attendees import dump_attendees, load_attendees, normalize_attendees

        names = ["Ada", "ada", " Bob ", "", 12, "Carol"]
        cleaned = normalize_attendees(names)
        self.assertEqual(cleaned, ["Ada", "Bob", "Carol"])
        raw = dump_attendees(names)
        self.assertEqual(json.loads(raw), ["Ada", "Bob", "Carol"])
        self.assertEqual(load_attendees(raw), ["Ada", "Bob", "Carol"])
        self.assertEqual(load_attendees(["X", "X"]), ["X"])
        self.assertEqual(load_attendees(None), [])

    def test_type_decorator_bind_result(self):
        from app.services.attendees import AttendeesJSON

        col = AttendeesJSON()
        bound = col.process_bind_param(["Ann", "Ann", "Bee"], None)
        self.assertEqual(json.loads(bound), ["Ann", "Bee"])
        self.assertEqual(col.process_result_value(bound, None), ["Ann", "Bee"])


class SegmentTimesTests(unittest.TestCase):
    def test_coerce_either_key_pair(self):
        from app.services.segment_times import coerce_times, segment_wire_dict, segments_from_asr

        self.assertEqual(coerce_times({"start": 1.5, "end": 3.0}), (1.5, 3.0))
        self.assertEqual(
            coerce_times({"start_time": 2.0, "end_time": 4.0}), (2.0, 4.0)
        )
        # Prefer start_time when both present.
        self.assertEqual(
            coerce_times({"start": 9.0, "end": 9.5, "start_time": 1.0, "end_time": 2.0}),
            (1.0, 2.0),
        )
        wire = segment_wire_dict(text="hi", start=1.0, end=2.0, seq=3)
        self.assertEqual(wire["start"], 1.0)
        self.assertEqual(wire["start_time"], 1.0)
        self.assertEqual(wire["end"], 2.0)
        self.assertEqual(wire["end_time"], 2.0)

        class _Seg:
            text = "a"
            start = 0.5
            end = 1.5

        out = segments_from_asr([_Seg()])
        self.assertEqual(out[0]["start_time"], 0.5)
        self.assertEqual(out[0]["start"], 0.5)

    def test_absolute_window_times(self):
        from app.services.segment_times import absolute_window_times

        # 16 kHz int16 mono: 32000 bytes = 1.0s
        start, end = absolute_window_times(
            byte_offset=32000,
            sample_rate=16000,
            relative_start=0.25,
            relative_end=0.75,
        )
        self.assertAlmostEqual(start, 1.25)
        self.assertAlmostEqual(end, 1.75)


class TranscriptSegmentSchemaTests(unittest.TestCase):
    def test_response_exposes_wire_aliases(self):
        from app.schemas import TranscriptSegmentResponse

        seg = TranscriptSegmentResponse(
            id="1",
            kind="final",
            text="hello",
            start_time=1.25,
            end_time=2.5,
            seq=0,
        )
        dumped = seg.model_dump()
        self.assertEqual(dumped["start_time"], 1.25)
        self.assertEqual(dumped["start"], 1.25)
        self.assertEqual(dumped["end_time"], 2.5)
        self.assertEqual(dumped["end"], 2.5)


class AiQualityPersistenceTests(unittest.TestCase):
    def test_faithfulness_roundtrip(self):
        from app.services.ai_quality import dump_faithfulness, load_faithfulness

        report = {
            "status": "warn",
            "checked": 2,
            "untraced": [{"section": "Decisions", "line": "Launch Mars", "overlap": 0.0}],
        }
        raw = dump_faithfulness(report)
        self.assertTrue(raw)
        loaded = load_faithfulness(raw)
        self.assertEqual(loaded["status"], "warn")
        self.assertEqual(len(loaded["untraced"]), 1)


if __name__ == "__main__":
    unittest.main()
