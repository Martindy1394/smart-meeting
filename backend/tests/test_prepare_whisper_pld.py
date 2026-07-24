"""Tests for Whisper PLD cleaning / speaker-disjoint splits."""
from __future__ import annotations

import json
import struct
import sys
import tempfile
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "hiligaynon_asr"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prepare_whisper_pld as prep  # noqa: E402


def _write_wav(path: Path, *, seconds: float = 2.0, rate: int = 16000) -> None:
    n = int(seconds * rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        # Silent PCM
        wf.writeframes(b"\x00\x00" * n)


def _mini_pld(tmp: Path) -> Path:
    """Build a tiny multi-speaker HIL tree with read + spontaneous rows."""
    lang = tmp / "HIL"
    for speaker, texts in {
        "0001": [
            ('utt_a.wav', "news_read", "Maayong aga sa inyo tanan."),
            ('utt_b.wav', "spontaneous_q1", "Ano ang imo ginahimo subong?"),
        ],
        "0002": [
            ('utt_c.wav', "medical_read", "Indi ko gid makita ang document (draft)."),
            ('utt_d.wav', "tourism_read", "Ang Plaza Libertad ara sa Iloilo."),
        ],
        "0003": [
            ('utt_e.wav', "education_read", "Nagtuon kami sang Hiligaynon kag English."),
        ],
        "0004": [
            ('utt_f.wav', "news_read", "Wala sang budget para sa proyekto."),
        ],
    }.items():
        speaker_dir = lang / speaker
        lines = [f'SpeakerID = "{speaker}"']
        for wav_name, prompt, text in texts:
            _write_wav(speaker_dir / wav_name, seconds=2.5)
            lines.append(f'{wav_name} "{prompt}" "{text}"')
        # One too-short clip
        if speaker == "0001":
            _write_wav(speaker_dir / "short.wav", seconds=0.2)
            lines.append('short.wav "news_read" "Short clip."')
        (speaker_dir / "session.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lang


def test_filters_drop_spontaneous_digits_short():
    with tempfile.TemporaryDirectory(prefix="pld_clean_") as td:
        lang = _mini_pld(Path(td))
        # Import raw via prepare's dependency path
        import import_pld

        raw = import_pld.import_pld_language(lang, language="hil")
        cleaned, rejections = prep.clean_rows(
            raw,
            min_duration=1.0,
            max_duration=15.0,
            keep_spontaneous=False,
            keep_digits_parens=False,
        )
        texts = {r["text"] for r in cleaned}
        assert "Ano ang imo ginahimo subong?" not in texts  # spontaneous
        assert not any("(" in t for t in texts)  # digits/parens row dropped
        assert "Short clip." not in texts
        assert rejections["spontaneous_source"] >= 1
        assert rejections["parentheses_or_digits"] >= 1
        assert rejections["too_short"] >= 1
        assert len(cleaned) >= 3


def test_speaker_disjoint_splits():
    rows = [
        {"audio": f"/x/{i}.wav", "text": "a", "language": "hil", "speaker_id": sid, "id": f"{i}"}
        for i, sid in enumerate(["s1", "s1", "s2", "s2", "s3", "s4", "s5"] * 2)
    ]
    splits = prep.speaker_disjoint_splits(rows, dev_ratio=0.2, test_ratio=0.2, seed=0)
    train_sp = {r["speaker_id"] for r in splits["train"]}
    dev_sp = {r["speaker_id"] for r in splits["dev"]}
    test_sp = {r["speaker_id"] for r in splits["test"]}
    assert train_sp.isdisjoint(dev_sp)
    assert train_sp.isdisjoint(test_sp)
    assert dev_sp.isdisjoint(test_sp)


def test_cli_writes_splits():
    with tempfile.TemporaryDirectory(prefix="pld_prep_cli_") as td:
        base = Path(td)
        lang = _mini_pld(base)
        out = base / "clean"
        rc = prep.main(
            [
                "--pld-lang-dir",
                str(lang),
                "--language",
                "hil",
                "--out-dir",
                str(out),
            ]
        )
        assert rc == 0
        assert (out / "train.jsonl").exists()
        assert (out / "summary.json").exists()
        train = [
            json.loads(line)
            for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert train
        assert "audio" in train[0] and "text" in train[0]


if __name__ == "__main__":
    test_filters_drop_spontaneous_digits_short()
    test_speaker_disjoint_splits()
    test_cli_writes_splits()
    print("all_prepare_whisper_pld_tests_passed")
