#!/usr/bin/env python3
"""Build a JSONL fine-tune dataset from WAV + matching transcript files.

Expected layout:
  data/
    clip001.wav
    clip001.txt
    clip002.wav
    clip002.txt

Or a CSV with columns: audio,text

For UP-DSP Philippine Languages Database (PLD) session folders, use
``import_pld.py`` instead (preferred Hiligaynon source — see docs/PLD.md).

Output JSONL rows: {"audio": "...", "text": "...", "language": "hil"}

Usage:
  python scripts/hiligaynon_asr/prepare_dataset.py \\
    --input-dir ./hil-data --output ./hil-train.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def pairs_from_dir(root: Path) -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    for audio in sorted(root.rglob("*")):
        if audio.suffix.lower() not in AUDIO_EXTS:
            continue
        txt = audio.with_suffix(".txt")
        if not txt.exists():
            print(f"skip (no transcript): {audio}", file=sys.stderr)
            continue
        text = txt.read_text(encoding="utf-8").strip()
        if not text:
            print(f"skip (empty transcript): {audio}", file=sys.stderr)
            continue
        out.append((audio.resolve(), text))
    return out


def pairs_from_csv(path: Path) -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio = Path(row.get("audio") or row.get("path") or "").expanduser()
            text = (row.get("text") or row.get("transcript") or "").strip()
            if not audio.exists() or not text:
                continue
            out.append((audio.resolve(), text))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, help="Directory of audio+txt pairs")
    p.add_argument("--csv", type=Path, help="CSV with audio,text columns")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--language", default="hil")
    args = p.parse_args(argv)

    pairs: list[tuple[Path, str]] = []
    if args.input_dir:
        pairs.extend(pairs_from_dir(args.input_dir))
    if args.csv:
        pairs.extend(pairs_from_csv(args.csv))
    if not pairs:
        print("No audio/transcript pairs found.", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for audio, text in pairs:
            f.write(
                json.dumps(
                    {"audio": str(audio), "text": text, "language": args.language},
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"wrote {len(pairs)} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
