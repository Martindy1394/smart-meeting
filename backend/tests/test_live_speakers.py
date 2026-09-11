"""Live Voice 1 / Voice 2 clustering (no extra ML library)."""
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


if __name__ == "__main__":
    unittest.main()
