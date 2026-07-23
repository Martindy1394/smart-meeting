#!/usr/bin/env python3
"""Import UP-DSP Philippine Languages Database (PLD) sessions into JSONL.

PLD (Guevara et al., SIGUL 2024) is a multilingual speech corpus (~454h) covering
Filipino, English, Cebuano, Kapampangan, Hiligaynon, Ilokano, Bikolano, Waray,
Tausug, and Pangasinense. Smart Meeting uses the Hiligaynon subset (~41h) as the
recommended fine-tune source for Ilonggo ASR.

Expected PLD layout (per language folder)::

  PLD/HIL/
    0123/
      session.log
      utterance_001.wav
      ...

Log lines (from the UP recording tool)::

  SpeakerID = "0123"
  SpeakerGender = "F"
  utterance_001.wav "unused" "Ang text sang transcript."

Output JSONL rows match ``finetune_whisper.py`` / ``prepare_dataset.py``::

  {"audio": "/abs/path.wav", "text": "...", "language": "hil", "speaker_id": "..."}

Obtain PLD from Mozilla Data Collective (UP-DSP-PLD, CC-BY-NC-4.0 research use):
  https://mozilladatacollective.com/datasets/cmmxhw46c00tqnw07xyr94zjk
Paper: https://aclanthology.org/2024.sigul-1.32/

Usage::

  python scripts/hiligaynon_asr/import_pld.py \\
    --pld-lang-dir ./data/PLD/HIL \\
    --output ./hil-pld-train.jsonl \\
    --language hil

  # Or point at the PLD root and pick a language code:
  python scripts/hiligaynon_asr/import_pld.py \\
    --pld-root ./data/PLD --language hil --output ./hil-pld-train.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# UP recording-tool log formats (same as dka-speech PLD adapter).
LOG_ROW = re.compile(r'^(?P<wav>\S+\.wav)\s+"[^"]+"\s+"(?P<text>.*)"\s*$')
META_ROW = re.compile(r'^(?P<key>\w+)\s*=\s+"?(?P<value>.*?)"?\s*$')

# PLD language folder aliases → app language code.
PLD_LANGUAGE_FOLDERS: dict[str, str] = {
    "hil": "hil",
    "hiligaynon": "hil",
    "ilonggo": "hil",
    "ceb": "ceb",
    "cebuano": "ceb",
    "bisaya": "ceb",
    "fil": "fil",
    "filipino": "fil",
    "tl": "tl",
    "tagalog": "tl",
    "ilo": "ilo",
    "ilokano": "ilo",
    "ilocano": "ilo",
    "bik": "bik",
    "bikol": "bik",
    "bikolano": "bik",
    "war": "war",
    "waray": "war",
    "pam": "pam",
    "kapampangan": "pam",
    "pag": "pag",
    "pangasinense": "pag",
    "pangasinan": "pag",
    "tsg": "tsg",
    "tausug": "tsg",
    "eng": "en",
    "en": "en",
    "english": "en",
}


def normalize_pld_language(code: str) -> str:
    key = (code or "").strip().lower()
    return PLD_LANGUAGE_FOLDERS.get(key, key or "hil")


def read_meta(log_path: Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = META_ROW.match(line.strip())
        if match:
            meta[match.group("key")] = match.group("value")
    return meta


def read_utterances(log_path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = LOG_ROW.match(line.strip())
        if match:
            text = match.group("text").strip()
            if text:
                rows.append((match.group("wav"), text))
    return rows


def resolve_lang_dir(pld_root: Path | None, pld_lang_dir: Path | None, language: str) -> Path:
    if pld_lang_dir is not None:
        return pld_lang_dir
    if pld_root is None:
        raise ValueError("Provide --pld-lang-dir or --pld-root")
    code = normalize_pld_language(language)
    # Try common folder spellings used in PLD distributions.
    candidates = [
        pld_root / code.upper(),
        pld_root / code,
        pld_root / language.upper(),
        pld_root / language,
    ]
    # Hiligaynon sometimes shipped as HILIGAYNON /
    if code == "hil":
        candidates.extend(
            [
                pld_root / "HILIGAYNON",
                pld_root / "Hiligaynon",
                pld_root / "ILONGGO",
            ]
        )
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError(
        f"No PLD language folder for '{language}' under {pld_root}. "
        f"Tried: {', '.join(str(c) for c in candidates)}"
    )


def import_pld_language(
    lang_dir: Path,
    *,
    language: str,
    limit: int | None = None,
) -> list[dict]:
    """Parse all speaker session logs under a PLD language directory."""
    lang = normalize_pld_language(language)
    rows: list[dict] = []
    for log_path in sorted(lang_dir.glob("*/*.log")):
        meta = read_meta(log_path)
        speaker_id = meta.get("SpeakerID") or log_path.parent.name
        for wav_name, text in read_utterances(log_path):
            wav_path = (log_path.parent / wav_name).resolve()
            if not wav_path.is_file():
                continue
            rows.append(
                {
                    "audio": str(wav_path),
                    "text": text,
                    "language": lang,
                    "speaker_id": speaker_id,
                    "gender": meta.get("SpeakerGender", ""),
                    "age": meta.get("SpeakerAge", ""),
                    "dialect": meta.get("SpeakerDialect", ""),
                    "source": "UP-DSP-PLD",
                }
            )
            if limit is not None and len(rows) >= limit:
                return rows
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pld-lang-dir", type=Path, help="Path to one PLD language folder (e.g. PLD/HIL)")
    p.add_argument("--pld-root", type=Path, help="Path to PLD root containing language folders")
    p.add_argument("--language", default="hil", help="PLD language code (default: hil)")
    p.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    p.add_argument("--limit", type=int, default=None, help="Max utterances (for smoke tests)")
    args = p.parse_args(argv)

    try:
        lang_dir = resolve_lang_dir(args.pld_root, args.pld_lang_dir, args.language)
    except (ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    rows = import_pld_language(lang_dir, language=args.language, limit=args.limit)
    if not rows:
        print(f"No PLD utterances found under {lang_dir}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    hours_hint = len(rows) * 4.7 / 3600.0  # PLD average ~4.7s/utt
    print(f"wrote {len(rows)} utterances (~{hours_hint:.1f}h est.) -> {args.output}")
    print(f"language={normalize_pld_language(args.language)} source={lang_dir}")
    print("Next:")
    print(
        f"  python scripts/hiligaynon_asr/finetune_whisper.py "
        f"--train-jsonl {args.output} "
        f"--output-dir ./models/whisper-medium-pld-{normalize_pld_language(args.language)} "
        f"--model-name openai/whisper-medium --fp16"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
