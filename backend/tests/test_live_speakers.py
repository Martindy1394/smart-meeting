"""Live Voice labels: cluster talkers, then rank Voice 1 by ASR accuracy."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services import live_speakers


def _tone(hz: float, seconds: float = 0.8, sr: int = 16000, amp: float = 0.25) -> bytes:
    t = np.arange(int(sr * seconds), dtype=np.float32) / sr
    # Add a weak harmonic so two pitches have distinct spectral envelopes.
    wave = np.sin(2 * np.pi * hz * t) * amp + np.sin(2 * np.pi * (2 * hz) * t) * (amp * 0.35)
    return (np.clip(wave, -1, 1) * 32767).astype("<i2").tobytes()


class LiveSpeakerTests(unittest.TestCase):
    def test_voice_label_format(self):
        self.assertEqual(live_speakers.voice_label(1), "Voice 1")
        self.assertEqual(live_speakers.voice_label(3), "Voice 3")

    def test_distinct_pitch_windows_get_different_voices(self):
        live_speakers.reset_meeting("meet-voices")
        low = _tone(90.0)
        high = _tone(280.0)
        i1, l1 = live_speakers.label_pcm("meet-voices", low)
        i2, l2 = live_speakers.label_pcm("meet-voices", high)
        self.assertTrue(l1.startswith("Voice"))
        self.assertTrue(l2.startswith("Voice"))
        self.assertNotEqual(i1, i2)
        self.assertTrue({i1, i2} <= {1, 2, 3})

    def test_similar_windows_reuse_same_voice(self):
        live_speakers.reset_meeting("meet-same")
        a = _tone(140.0)
        b = _tone(141.0)
        i1, _ = live_speakers.label_pcm("meet-same", a)
        i2, _ = live_speakers.label_pcm("meet-same", b)
        self.assertEqual(i1, i2)
        self.assertEqual(i1, 1)

    def test_format_transcript_groups_turns(self):
        class Seg:
            def __init__(self, text, label):
                self.text = text
                self.speaker_label = label

        text = live_speakers.format_transcript(
            [
                Seg("Hello board.", "Voice 1"),
                Seg("We should vote.", "Voice 1"),
                Seg("I second that.", "Voice 2"),
            ]
        )
        self.assertIn("Voice 1: Hello board. We should vote.", text)
        self.assertIn("Voice 2: I second that.", text)

    def test_prefix_text_idempotent(self):
        self.assertEqual(
            live_speakers.prefix_text("Voice 1", "Voice 1: hi"), "Voice 1: hi"
        )
        self.assertEqual(live_speakers.prefix_text("Voice 1", "hi"), "Voice 1: hi")

    def test_label_segments_uses_audio_slices(self):
        sr = 16000
        live_speakers.reset_meeting("meet-slices")
        low = np.frombuffer(_tone(90.0, seconds=1.0), dtype="<i2").astype(np.float32) / 32768.0
        high = np.frombuffer(_tone(280.0, seconds=1.0), dtype="<i2").astype(np.float32) / 32768.0
        wav = np.concatenate([low, high])

        class Seg:
            def __init__(self, text, start, end):
                self.text = text
                self.start = start
                self.end = end
                self.speaker_label = ""
                self.speaker_index = 0

        segs = [
            Seg("hello", 0.0, 1.0),
            Seg("board", 1.0, 2.0),
        ]
        out = live_speakers.label_segments("meet-slices", segs, wav, sample_rate=sr)
        self.assertTrue(out[0].speaker_label.startswith("Voice"))
        self.assertTrue(out[1].speaker_label.startswith("Voice"))
        self.assertNotEqual(out[0].speaker_index, out[1].speaker_index)

    def test_labels_rank_by_asr_accuracy(self):
        sr = 16000
        live_speakers.reset_meeting("meet-acc")
        low = np.frombuffer(_tone(90.0, seconds=1.0), dtype="<i2").astype(np.float32) / 32768.0
        high = np.frombuffer(_tone(280.0, seconds=1.0), dtype="<i2").astype(np.float32) / 32768.0
        wav = np.concatenate([low, high])

        class Seg:
            def __init__(self, text, start, end, avg_logprob, no_speech_prob=0.1):
                self.text = text
                self.start = start
                self.end = end
                self.avg_logprob = avg_logprob
                self.no_speech_prob = no_speech_prob
                self.low_confidence = False
                self.speaker_label = ""
                self.speaker_index = 0

        # First talker is less accurate; second talker should become Voice 1.
        segs = [
            Seg("hello", 0.0, 1.0, avg_logprob=-0.9),
            Seg("board", 1.0, 2.0, avg_logprob=-0.1),
        ]
        out = live_speakers.label_segments("meet-acc", segs, wav, sample_rate=sr)
        self.assertEqual(out[1].speaker_label, "Voice 1")
        self.assertEqual(out[0].speaker_label, "Voice 2")
        self.assertEqual(out[1].speaker_index, 1)
        self.assertEqual(out[0].speaker_index, 2)

    def test_bind_asr_accuracy_promotes_best_cluster(self):
        live_speakers.reset_meeting("meet-live-acc")
        low = _tone(90.0)
        high = _tone(280.0)
        i1, _ = live_speakers.label_pcm("meet-live-acc", low)
        i2, _ = live_speakers.label_pcm("meet-live-acc", high)
        self.assertNotEqual(i1, i2)

        class Result:
            def __init__(self, lp):
                self.segments = [type("S", (), {"avg_logprob": lp, "no_speech_prob": 0.05, "low_confidence": False})()]
                self.language_confidence = None

        live_speakers.bind_asr_accuracy("meet-live-acc", i1, Result(-0.95))
        live_speakers.bind_asr_accuracy("meet-live-acc", i2, Result(-0.05))
        d1, l1 = live_speakers.bind_asr_accuracy("meet-live-acc", i1)
        d2, l2 = live_speakers.bind_asr_accuracy("meet-live-acc", i2)
        self.assertEqual(l2, "Voice 1")
        self.assertEqual(d2, 1)
        self.assertEqual(l1, "Voice 2")
        self.assertEqual(d1, 2)

    def test_rank_without_scores_keeps_first_seen_order(self):
        mapping = live_speakers.rank_voice_ids([1, 2, 1], {})
        self.assertEqual(mapping, {1: 1, 2: 2})

    def test_registered_speaker_count(self):
        from types import SimpleNamespace

        self.assertEqual(
            live_speakers.registered_speaker_count(
                attendees=["Ada", "Bob"], presiding_officer="Chair"
            ),
            3,
        )
        self.assertEqual(
            live_speakers.registered_speaker_count(
                SimpleNamespace(attendees=["Ada"], presiding_officer="")
            ),
            1,
        )
        self.assertEqual(
            live_speakers.registered_speaker_count(attendees=[], presiding_officer=""),
            1,
        )

    def test_participant_count_caps_voice_slots(self):
        sr = 16000
        low = np.frombuffer(_tone(90.0, seconds=1.0), dtype="<i2").astype(np.float32) / 32768.0
        high = np.frombuffer(_tone(280.0, seconds=1.0), dtype="<i2").astype(np.float32) / 32768.0
        wav = np.concatenate([low, high])

        class Seg:
            def __init__(self, text, start, end):
                self.text = text
                self.start = start
                self.end = end
                self.speaker_label = ""
                self.speaker_index = 0

        segs = [Seg("hello", 0.0, 1.0), Seg("board", 1.0, 2.0)]
        out = live_speakers.label_segments(
            "meet-one-slot", segs, wav, sample_rate=sr, max_voices=1
        )
        self.assertEqual(out[0].speaker_index, 1)
        self.assertEqual(out[1].speaker_index, 1)
        self.assertEqual(out[0].speaker_label, "Voice 1")
        self.assertEqual(out[1].speaker_label, "Voice 1")

    def test_clamp_voice_index_keeps_extra_whisper_voices(self):
        self.assertEqual(live_speakers.clamp_voice_index(7, registered_slots=2), 7)
        self.assertEqual(live_speakers.clamp_voice_index(99, registered_slots=1), 32)
        self.assertEqual(live_speakers.clamp_voice_index(0), 1)
        self.assertEqual(live_speakers.voice_label(7), "Voice 7")

    def test_registered_count_null_attendees(self):
        from types import SimpleNamespace

        self.assertEqual(
            live_speakers.registered_speaker_count(
                SimpleNamespace(attendees=None, presiding_officer=None)
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
