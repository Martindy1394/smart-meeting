"""WER helper tests (no model weights)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "hiligaynon_asr"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import wer  # noqa: E402


def test_perfect_match_zero_wer():
    m = wer.word_error_rate("ang mga bata", "ang mga bata")
    assert m["wer"] == 0.0
    assert m["edits"] == 0


def test_substitution_counts():
    m = wer.word_error_rate("one two three", "one too three")
    assert m["edits"] == 1
    assert abs(m["wer"] - (1 / 3)) < 1e-9


def test_normalize_strips_punctuation():
    assert wer.normalize_text("Hello, World!") == "hello world"


if __name__ == "__main__":
    test_perfect_match_zero_wer()
    test_substitution_counts()
    test_normalize_strips_punctuation()
    print("all_wer_tests_passed")
