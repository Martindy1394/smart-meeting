"""Unit tests for UP-DSP-PLD log import (no full corpus required)."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "hiligaynon_asr"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import import_pld  # noqa: E402


def _write_mini_pld(tmp: Path) -> Path:
    lang = tmp / "HIL" / "0123"
    lang.mkdir(parents=True)
    (lang / "utt_001.wav").write_bytes(b"RIFF....WAVE")
    (lang / "session.log").write_text(
        'SpeakerID = "0123"\n'
        'SpeakerGender = "F"\n'
        'SpeakerAge = "24"\n'
        'SpeakerDialect = "Iloilo"\n'
        'utt_001.wav "prompt" "Wala gid kami sang budget subong."\n'
        'missing.wav "prompt" "This wav is absent."\n',
        encoding="utf-8",
    )
    return tmp / "HIL"


def test_normalize_pld_language():
    assert import_pld.normalize_pld_language("HIL") == "hil"
    assert import_pld.normalize_pld_language("Hiligaynon") == "hil"
    assert import_pld.normalize_pld_language("ceb") == "ceb"
    assert import_pld.normalize_pld_language("Tausug") == "tsg"


def test_import_pld_language_reads_log():
    with tempfile.TemporaryDirectory(prefix="pld_test_") as td:
        lang_dir = _write_mini_pld(Path(td))
        rows = import_pld.import_pld_language(lang_dir, language="hil")
        assert len(rows) == 1
        assert rows[0]["language"] == "hil"
        assert rows[0]["speaker_id"] == "0123"
        assert "wala gid" in rows[0]["text"].lower()
        assert rows[0]["source"] == "UP-DSP-PLD"
        assert Path(rows[0]["audio"]).name == "utt_001.wav"


def test_cli_writes_jsonl():
    with tempfile.TemporaryDirectory(prefix="pld_cli_") as td:
        base = Path(td)
        lang_dir = _write_mini_pld(base)
        out = base / "out.jsonl"
        rc = import_pld.main(
            [
                "--pld-lang-dir",
                str(lang_dir),
                "--output",
                str(out),
                "--language",
                "hil",
            ]
        )
        assert rc == 0
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row["language"] == "hil"


if __name__ == "__main__":
    test_normalize_pld_language()
    test_import_pld_language_reads_log()
    test_cli_writes_jsonl()
    print("all_pld_import_tests_passed")
