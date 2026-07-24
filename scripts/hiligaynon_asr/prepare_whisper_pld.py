#!/usr/bin/env python3
"""Clean UP-DSP PLD audio for Whisper fine-tuning (Smart Meeting ASR).

Ports the filtering / speaker-disjoint split ideas from the OmniVoice Filipino
PLD prep script to this project's Whisper JSONL format:

* Drop spontaneous prompt sources
* Drop transcripts with parentheses or digits (noisy / non-speech markup)
* Drop empty / unreadable / out-of-range duration clips (default 1–15s)
* Speaker-disjoint train / dev / test splits
* Optional hardlink/copy package under ``--package-dir``

Usage::

  # Hiligaynon (recommended for Smart Meeting)
  python3 scripts/hiligaynon_asr/prepare_whisper_pld.py \\
    --pld-root ./data/PLD --language hil \\
    --out-dir ./data/pld_hiligaynon_clean

  # Then fine-tune:
  python3 scripts/hiligaynon_asr/finetune_whisper.py \\
    --train-jsonl ./data/pld_hiligaynon_clean/train.jsonl \\
    --eval-jsonl ./data/pld_hiligaynon_clean/dev.jsonl \\
    --output-dir ./models/whisper-medium-pld-hil \\
    --model-name openai/whisper-medium --fp16

Windows / Git Bash (one line)::

  python3 scripts/hiligaynon_asr/prepare_whisper_pld.py --pld-root "M:/MSCS/PLD" --language hil --out-dir ./data/pld_hiligaynon_clean
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Reuse PLD path / log helpers from the importer.
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import import_pld  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pld-root", type=Path, help="PLD root (or parent of HIL/FIL/…)")
    p.add_argument("--pld-lang-dir", type=Path, help="Direct language folder (e.g. …/HIL)")
    p.add_argument("--language", default="hil", help="PLD language code (default: hil)")
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for train/dev/test JSONL + summary.json",
    )
    p.add_argument("--min-duration", type=float, default=1.0)
    p.add_argument("--max-duration", type=float, default=15.0)
    p.add_argument("--dev-ratio", type=float, default=0.10)
    p.add_argument("--test-ratio", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--keep-spontaneous",
        action="store_true",
        help="Keep spontaneous / _spo_ prompt sources (excluded by default)",
    )
    p.add_argument(
        "--keep-digits-parens",
        action="store_true",
        help="Keep transcripts that contain digits or parentheses",
    )
    p.add_argument(
        "--package-dir",
        type=Path,
        default=None,
        help="Optional standalone package with wavs/ + relative JSONL paths",
    )
    p.add_argument(
        "--package-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="How to materialize WAVs in --package-dir (default: hardlink)",
    )
    p.add_argument("--limit", type=int, default=None, help="Max raw rows before filters")
    return p.parse_args(argv)


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
        if rate <= 0:
            raise ValueError("invalid sample rate")
        return frames / float(rate)


def is_spontaneous_source(prompt_source: str) -> bool:
    lowered = (prompt_source or "").lower()
    return "spontaneous" in lowered or "_spo_" in lowered


def has_parentheses_or_digits(text: str) -> bool:
    return "(" in text or ")" in text or any(ch.isdigit() for ch in text)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    duration = sum(float(row.get("duration") or 0.0) for row in rows)
    speakers = {row.get("speaker_id") for row in rows}
    return {
        "samples": len(rows),
        "hours": round(duration / 3600.0, 3),
        "speakers": len(speakers),
    }


def clean_rows(
    raw_rows: list[dict[str, Any]],
    *,
    min_duration: float,
    max_duration: float,
    keep_spontaneous: bool,
    keep_digits_parens: bool,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    cleaned: list[dict[str, Any]] = []
    rejections: Counter[str] = Counter()

    for row in raw_rows:
        audio = Path(row["audio"])
        text = " ".join((row.get("text") or "").split())
        prompt = row.get("prompt_source") or ""

        if not audio.is_file():
            rejections["missing_wav"] += 1
            continue
        if audio.stat().st_size == 0:
            rejections["zero_byte_wav"] += 1
            continue
        if not text:
            rejections["empty_text"] += 1
            continue
        if not keep_spontaneous and is_spontaneous_source(prompt):
            rejections["spontaneous_source"] += 1
            continue
        if not keep_digits_parens and has_parentheses_or_digits(text):
            rejections["parentheses_or_digits"] += 1
            continue
        try:
            duration = wav_duration_seconds(audio)
        except Exception:
            rejections["unreadable_wav"] += 1
            continue
        if duration < min_duration:
            rejections["too_short"] += 1
            continue
        if duration > max_duration:
            rejections["too_long"] += 1
            continue

        cleaned.append(
            {
                "audio": str(audio.resolve()),
                "text": text,
                "language": row.get("language") or "hil",
                "speaker_id": str(row.get("speaker_id") or ""),
                "prompt_source": prompt,
                "duration": round(duration, 6),
                "corpus": row.get("corpus") or "UP-DSP-PLD",
                "id": audio.stem,
            }
        )
    return cleaned, rejections


def speaker_disjoint_splits(
    rows: list[dict[str, Any]],
    *,
    dev_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_speaker[str(row.get("speaker_id") or "unknown")].append(row)

    speaker_ids = sorted(by_speaker)
    rng = random.Random(seed)
    rng.shuffle(speaker_ids)

    if len(speaker_ids) < 3:
        # Tiny fixtures: put everything in train, leave empty eval if needed.
        return {"train": list(rows), "dev": [], "test": []}

    n = len(speaker_ids)
    n_dev = max(1, round(n * dev_ratio))
    n_test = max(1, round(n * test_ratio))
    # Keep at least one train speaker.
    while n_dev + n_test >= n and (n_dev > 1 or n_test > 1):
        if n_dev >= n_test and n_dev > 1:
            n_dev -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break

    dev_speakers = set(speaker_ids[:n_dev])
    test_speakers = set(speaker_ids[n_dev : n_dev + n_test])
    train_speakers = set(speaker_ids[n_dev + n_test :])

    def pick(speakers: set[str]) -> list[dict[str, Any]]:
        out = [row for row in rows if str(row.get("speaker_id")) in speakers]
        return sorted(out, key=lambda r: r.get("id") or r["audio"])

    return {
        "train": pick(train_speakers),
        "dev": pick(dev_speakers),
        "test": pick(test_speakers),
    }


def materialize_package(
    package_dir: Path,
    splits: dict[str, list[dict[str, Any]]],
    *,
    mode: str,
) -> dict[str, Any]:
    wav_dir = package_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    linked = 0
    reused = 0
    all_rows = [row for rows in splits.values() for row in rows]

    for row in all_rows:
        source = Path(row["audio"])
        speaker = str(row.get("speaker_id") or "unknown")
        target = wav_dir / speaker / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.stat().st_size != source.stat().st_size:
                raise RuntimeError(f"Existing packaged file has wrong size: {target}")
            reused += 1
        else:
            if mode == "hardlink":
                try:
                    os.link(source, target)
                except OSError:
                    # Cross-device / Windows privilege fallback.
                    shutil.copy2(source, target)
            else:
                shutil.copy2(source, target)
            linked += 1

    packaged: dict[str, list[dict[str, Any]]] = {}
    for split, rows in splits.items():
        packaged_rows = []
        for row in rows:
            source = Path(row["audio"])
            speaker = str(row.get("speaker_id") or "unknown")
            rel = Path("wavs") / speaker / source.name
            packaged_rows.append(
                {
                    "audio": rel.as_posix(),
                    "text": row["text"],
                    "language": row["language"],
                    "speaker_id": row.get("speaker_id", ""),
                    "id": row.get("id", source.stem),
                }
            )
        packaged[split] = packaged_rows
        write_jsonl(package_dir / f"{split}.jsonl", packaged_rows)

    return {
        "package_dir": str(package_dir.resolve()),
        "materialization_mode": mode,
        "files_materialized": linked,
        "files_reused_existing": reused,
        "manifests": ["train.jsonl", "dev.jsonl", "test.jsonl"],
    }


def whisper_train_row(row: dict[str, Any]) -> dict[str, Any]:
    """Minimal row consumed by ``finetune_whisper.py``."""
    return {
        "audio": row["audio"],
        "text": row["text"],
        "language": row.get("language") or "hil",
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        lang_dir = import_pld.resolve_lang_dir(
            args.pld_root, args.pld_lang_dir, args.language
        )
    except (ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"using language dir: {lang_dir}", file=sys.stderr)
    try:
        raw_rows = import_pld.import_pld_language(
            lang_dir, language=args.language, limit=args.limit
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    cleaned, rejections = clean_rows(
        raw_rows,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        keep_spontaneous=args.keep_spontaneous,
        keep_digits_parens=args.keep_digits_parens,
    )
    if not cleaned:
        print(
            "No utterances left after cleaning. "
            f"raw={len(raw_rows)} rejections={dict(rejections)}",
            file=sys.stderr,
        )
        return 1

    splits = speaker_disjoint_splits(
        cleaned,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        write_jsonl(out_dir / f"{name}.jsonl", [whisper_train_row(r) for r in rows])
    write_jsonl(out_dir / "all_clean.jsonl", [whisper_train_row(r) for r in cleaned])

    summary: dict[str, Any] = {
        "language": import_pld.normalize_pld_language(args.language),
        "lang_dir": str(lang_dir.resolve()),
        "raw_rows": len(raw_rows),
        "filters": {
            "min_duration": args.min_duration,
            "max_duration": args.max_duration,
            "drop_spontaneous": not args.keep_spontaneous,
            "drop_digits_parens": not args.keep_digits_parens,
        },
        "cleaned": summarize_rows(cleaned),
        "splits": {name: summarize_rows(rows) for name, rows in splits.items()},
        "rejections": dict(rejections),
    }

    if args.package_dir:
        package_info = materialize_package(
            args.package_dir, splits, mode=args.package_mode
        )
        summary["package"] = package_info

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    readme = f"""# Cleaned PLD Whisper dataset ({summary['language']})

Generated by ``scripts/hiligaynon_asr/prepare_whisper_pld.py``.

## Files
- `train.jsonl` / `dev.jsonl` / `test.jsonl` — speaker-disjoint splits
- `all_clean.jsonl` — all cleaned rows
- `summary.json` — counts and rejection reasons

## Fine-tune Smart Meeting ASR
```bash
python3 scripts/hiligaynon_asr/finetune_whisper.py \\
  --train-jsonl {out_dir.as_posix()}/train.jsonl \\
  --eval-jsonl {out_dir.as_posix()}/dev.jsonl \\
  --output-dir ./models/whisper-medium-pld-{summary['language']} \\
  --model-name openai/whisper-medium --fp16
```

Then set ``WHISPER_HILIGAYNON_FINE_TUNED_MODEL`` (or Tagalog equivalent) to that
output directory.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(
        f"\nNext:\n  python3 scripts/hiligaynon_asr/finetune_whisper.py "
        f"--train-jsonl {out_dir / 'train.jsonl'} "
        f"--eval-jsonl {out_dir / 'dev.jsonl'} "
        f"--output-dir ./models/whisper-medium-pld-{summary['language']} "
        f"--model-name openai/whisper-medium --fp16"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
