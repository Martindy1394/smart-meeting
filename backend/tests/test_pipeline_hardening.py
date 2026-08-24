"""Tier 1–2 hardening: VAD, confidence, glossary, action items, translation faith."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


class VadGateTests(unittest.TestCase):
    def test_silence_skipped(self):
        from app.services import vad

        silence = np.zeros(16000, dtype=np.float32)
        result = vad.detect_speech(silence, live=True)
        self.assertFalse(result.has_speech)

    def test_speech_like_passes(self):
        from app.services import vad

        rng = np.random.default_rng(0)
        speech = (rng.normal(0, 0.12, 16000)).astype(np.float32)
        result = vad.detect_speech(speech, live=True)
        self.assertTrue(result.has_speech)


class SegmentConfidenceTests(unittest.TestCase):
    def test_high_no_speech_dropped(self):
        from types import SimpleNamespace

        from app.services.transcription import _segment_from_whisper

        seg = SimpleNamespace(
            start=0.0,
            end=1.0,
            avg_logprob=-0.2,
            no_speech_prob=0.95,
        )
        self.assertIsNone(_segment_from_whisper(seg, text="Thank you."))

    def test_low_confidence_flagged(self):
        from types import SimpleNamespace

        from app.services.transcription import _segment_from_whisper

        seg = SimpleNamespace(
            start=0.0,
            end=1.0,
            avg_logprob=-0.9,
            no_speech_prob=0.2,
        )
        out = _segment_from_whisper(seg, text="Maayong aga sa tanan.")
        self.assertIsNotNone(out)
        self.assertTrue(out.low_confidence)


class GlossaryTests(unittest.TestCase):
    def test_protect_restore_keeps_spelling(self):
        from app.services import glossary

        terms = glossary.load_glossary(["Iloilo City", "Garcia"])
        protected, mapping = glossary.protect(
            "Mayor Garcia visited Iloilo City yesterday.", terms
        )
        self.assertNotIn("Garcia", protected)
        restored = glossary.restore(
            protected.replace("⟦SMG", "⟦SMG"), mapping
        )
        # Restore original placeholders path
        restored = glossary.restore(protected, mapping)
        self.assertIn("Garcia", restored)
        self.assertIn("Iloilo City", restored)


class TranslationFaithfulnessTests(unittest.TestCase):
    def test_detects_missing_glossary_term(self):
        from app.services import llm

        report = llm.assess_translation_faithfulness(
            "Mayor Garcia spoke in Iloilo City.",
            "The mayor spoke in the capital.",
            glossary=["Garcia", "Iloilo City"],
        )
        self.assertEqual(report["status"], "warn")
        self.assertTrue(any(u["section"] == "Glossary" for u in report["untraced"]))


class ActionItemExtractionTests(unittest.TestCase):
    def test_extracts_owner_and_due(self):
        from app.services import action_items

        summary = (
            "Discussion\n• Budget review\n\n"
            "Action items\n"
            "• Maria will file the report by 15 March 2026\n"
            "• Prepare budget (Juan) — Friday\n"
        )
        items = action_items.extract_action_items(summary)
        self.assertGreaterEqual(len(items), 2)
        owners = {i.get("owner") for i in items}
        self.assertIn("Maria", owners)
        self.assertTrue(any(i.get("due_date") for i in items))


class CustomVocabPromptTests(unittest.TestCase):
    def test_prompt_appends_terms(self):
        from app.services.transcription import initial_prompt, parse_custom_vocab

        terms = parse_custom_vocab("Garcia\nIloilo City")
        prompt = initial_prompt("hil", extra_terms=terms)
        self.assertIsNotNone(prompt)
        self.assertIn("Garcia", prompt)
        self.assertIn("Iloilo City", prompt)


if __name__ == "__main__":
    unittest.main()
