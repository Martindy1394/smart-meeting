#!/usr/bin/env python3
"""Import UP-DSP Philippine Languages Database (PLD) sessions into JSONL.

PLD (Guevara et al., SIGUL 2024) is a multilingual speech corpus (~454h).
Smart Meeting uses the Hiligaynon subset (~41h) as the recommended fine-tune
source for Ilonggo ASR.

Expected layout (any nesting depth is OK)::

  …/PLD/HIL/<speaker_id>/*.wav + *.log

Log lines (UP recording tool)::

  SpeakerID = "0123"
  utterance_001.wav "unused" "Ang text sang transcript."

Usage (use ``python3``, not ``python``, on many Linux/macOS setups)::

  python3 scripts/hiligaynon_asr/import_pld.py \\
    --pld-root ./data/PLD --language hil --output ./hil-pld-train.jsonl

  # If you are unsure of the folder layout, inspect first:
  python3 scripts/hiligaynon_asr/import_pld.py --pld-root ./data --inspect

  # Or point directly at the Hiligaynon folder:
  python3 scripts/hiligaynon_asr/import_pld.py \\
    --pld-lang-dir ./data/PLD/HIL --output ./hil-pld-train.jsonl

Dataset: https://mozilladatacollective.com/datasets/cmmxhw46c00tqnw07xyr94zjk
Paper: https://aclanthology.org/2024.sigul-1.32/
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# UP recording-tool log formats (tolerant variants).
LOG_ROW = re.compile(
    r'^(?P<wav>\S+\.wav)\s+"[^"]*"\s+"(?P<text>.*)"\s*$',
    re.IGNORECASE,
)
# Some exports use: name.wav <tab> transcript
LOG_ROW_TSV = re.compile(
    r"^(?P<wav>\S+\.wav)\t+(?P<text>.+)$",
    re.IGNORECASE,
)
META_ROW = re.compile(r'^(?P<key>\w+)\s*=\s*"?(?P<value>.*?)"?\s*$')

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

# Folder names that mean Hiligaynon in common unpacks.
_HIL_DIR_NAMES = frozenset(
    {"hil", "hiligaynon", "ilonggo", "hiligaynon_ilonggo"}
)


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
    for raw in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = LOG_ROW.match(line) or LOG_ROW_TSV.match(line)
        if match:
            text = match.group("text").strip()
            if text:
                rows.append((match.group("wav"), text))
    return rows


def _list_top_dirs(root: Path, limit: int = 40) -> list[str]:
    if not root.is_dir():
        return []
    names = sorted(p.name for p in root.iterdir() if p.is_dir())
    if len(names) > limit:
        return names[:limit] + [f"… (+{len(names) - limit} more)"]
    return names


def find_language_dirs(root: Path, language: str) -> list[Path]:
    """Find language folders under root (any nesting, depth-limited)."""
    code = normalize_pld_language(language)
    wanted = {code, language.strip().lower()}
    if code == "hil":
        wanted |= _HIL_DIR_NAMES
    found: list[Path] = []
    if not root.is_dir():
        return found
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        if len(rel.parts) > 6:
            continue
        if path.name.casefold() in wanted:
            found.append(path)
    found = sorted(set(found), key=lambda p: (len(p.parts), str(p)))
    outermost: list[Path] = []
    for path in found:
        if any(path != prev and path.is_relative_to(prev) for prev in outermost):
            continue
        outermost.append(path)
    return outermost


def resolve_lang_dir(
    pld_root: Path | None, pld_lang_dir: Path | None, language: str
) -> Path:
    if pld_lang_dir is not None:
        path = pld_lang_dir.expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(
                f"--pld-lang-dir does not exist or is not a directory: {path}"
            )
        return path
    if pld_root is None:
        raise ValueError("Provide --pld-lang-dir or --pld-root")

    root = pld_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(
            f"--pld-root not found: {root}\n"
            "Unpack UP-DSP-PLD first, then point --pld-root at the folder that "
            "contains language dirs (HIL, CEB, …) or a parent of that folder.\n"
            "Example:\n"
            "  python3 scripts/hiligaynon_asr/import_pld.py "
            "--pld-root ./data/PLD --language hil --output ./hil-pld-train.jsonl\n"
            "Or inspect:\n"
            "  python3 scripts/hiligaynon_asr/import_pld.py "
            f"--pld-root {pld_root} --inspect"
        )
    if not root.is_dir():
        raise FileNotFoundError(f"--pld-root is not a directory: {root}")

    code = normalize_pld_language(language)
    # Direct children first (fast path).
    direct_candidates = [
        root / code.upper(),
        root / code,
        root / "PLD" / code.upper(),
        root / "PLD" / code,
        root / "UP-DSP-PLD" / "PLD" / code.upper(),
        root / "UP-DSP-PLD" / code.upper(),
    ]
    if code == "hil":
        for name in ("HILIGAYNON", "Hiligaynon", "ILONGGO", "hiligaynon"):
            direct_candidates.extend(
                [root / name, root / "PLD" / name, root / "UP-DSP-PLD" / "PLD" / name]
            )
    for path in direct_candidates:
        if path.is_dir():
            return path

    # Recursive discovery for zip-extract nesting.
    discovered = find_language_dirs(root, language)
    if discovered:
        return discovered[0]

    top = _list_top_dirs(root)
    raise FileNotFoundError(
        f"No PLD '{language}' (normalized={code}) language folder under {root}.\n"
        f"Top-level entries: {', '.join(top) if top else '(empty)'}\n"
        "Tips:\n"
        "  • Use python3 (not python) if you saw 'command not found'.\n"
        "  • Point --pld-root at the folder that contains HIL/ or Hiligaynon/,\n"
        "    or one level above it (importer searches nested folders).\n"
        "  • Or pass the language folder directly:\n"
        "      --pld-lang-dir /path/to/HIL\n"
        "  • Inspect the tree:\n"
        f"      python3 scripts/hiligaynon_asr/import_pld.py --pld-root {root} --inspect"
    )


def iter_log_files(lang_dir: Path) -> list[Path]:
    """Collect session logs under a language folder (any nesting)."""
    # Common: lang/speaker/*.log  — also accept deeper / flatter layouts.
    logs = sorted({*lang_dir.glob("*/*.log"), *lang_dir.glob("**/*.log")})
    return [p for p in logs if p.is_file()]


def import_pld_language(
    lang_dir: Path,
    *,
    language: str,
    limit: int | None = None,
) -> list[dict]:
    """Parse all speaker session logs under a PLD language directory."""
    lang = normalize_pld_language(language)
    rows: list[dict] = []
    logs = iter_log_files(lang_dir)
    missing_wav = 0
    parsed_utt = 0
    for log_path in logs:
        meta = read_meta(log_path)
        speaker_id = meta.get("SpeakerID") or log_path.parent.name
        utterances = read_utterances(log_path)
        parsed_utt += len(utterances)
        for wav_name, text in utterances:
            wav_path = (log_path.parent / wav_name).resolve()
            if not wav_path.is_file():
                # Some packs put wavs beside a nested transcript dir.
                alt = lang_dir / speaker_id / wav_name
                if alt.is_file():
                    wav_path = alt.resolve()
                else:
                    missing_wav += 1
                    continue
            rows.append(
                {
                    "audio": str(wav_path),
                    "text": text,
                    "language": lang,
                    "speaker_id": str(speaker_id),
                    "gender": meta.get("SpeakerGender", ""),
                    "age": meta.get("SpeakerAge", ""),
                    "dialect": meta.get("SpeakerDialect", ""),
                    "source": "UP-DSP-PLD",
                }
            )
            if limit is not None and len(rows) >= limit:
                return rows
    if not rows:
        # Attach diagnostics for the caller.
        raise RuntimeError(
            f"No usable PLD utterances under {lang_dir}\n"
            f"  log_files={len(logs)}  transcript_lines={parsed_utt}  "
            f"missing_wav={missing_wav}\n"
            "Expected files like: HIL/<speaker_id>/*.wav and matching *.log\n"
            "If logs use a different format, open one .log and share a few lines."
        )
    return rows


def inspect_tree(root: Path) -> int:
    root = root.expanduser().resolve()
    print(f"inspect root: {root}")
    if not root.exists():
        print("ERROR: path does not exist", file=sys.stderr)
        return 2
    if not root.is_dir():
        print("ERROR: path is not a directory", file=sys.stderr)
        return 2

    print(f"top-level: {', '.join(_list_top_dirs(root)) or '(empty)'}")
    for code in ("hil", "ceb", "fil", "ilo", "war", "bik", "pam", "pag", "tsg", "en"):
        hits = find_language_dirs(root, code)
        if hits:
            print(f"  [{code}] folders:")
            for hit in hits[:8]:
                logs = iter_log_files(hit)
                wavs = list(hit.glob("**/*.wav"))[:1]
                n_wav = sum(1 for _ in hit.glob("**/*.wav"))
                print(
                    f"    - {hit}  logs={len(logs)}  wavs≈{n_wav}"
                    + (f"  sample_wav={wavs[0].name}" if wavs else "")
                )
    # Also show any *.log nearby if no language folder matched.
    sample_logs = list(root.glob("**/*.log"))[:10]
    if sample_logs:
        print("sample .log files:")
        for log in sample_logs:
            print(f"  - {log}")
            # Preview first matching utterance line.
            for line in log.read_text(encoding="utf-8", errors="ignore").splitlines()[:30]:
                if LOG_ROW.match(line.strip()) or LOG_ROW_TSV.match(line.strip()):
                    print(f"      utt: {line.strip()[:120]}")
                    break
    else:
        print("No .log files found under this root (depth search).")
        print("If the download is still a .zip/.tar, unpack it first.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pld-lang-dir",
        type=Path,
        help="Path to one PLD language folder (e.g. …/PLD/HIL)",
    )
    p.add_argument(
        "--pld-root",
        type=Path,
        help="Path to PLD root (or a parent). Importer searches nested folders.",
    )
    p.add_argument("--language", default="hil", help="PLD language code (default: hil)")
    p.add_argument("--output", type=Path, help="Output JSONL path")
    p.add_argument("--limit", type=int, default=None, help="Max utterances (smoke test)")
    p.add_argument(
        "--inspect",
        action="store_true",
        help="Print discovered language folders / logs under --pld-root and exit",
    )
    args = p.parse_args(argv)

    if args.inspect:
        if args.pld_root is None and args.pld_lang_dir is None:
            print("Provide --pld-root (or --pld-lang-dir) with --inspect", file=sys.stderr)
            return 2
        return inspect_tree(args.pld_root or args.pld_lang_dir)

    if args.output is None:
        print("--output is required unless using --inspect", file=sys.stderr)
        return 2

    try:
        lang_dir = resolve_lang_dir(args.pld_root, args.pld_lang_dir, args.language)
    except (ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"using language dir: {lang_dir}", file=sys.stderr)
    try:
        rows = import_pld_language(lang_dir, language=args.language, limit=args.limit)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
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
        f"  python3 scripts/hiligaynon_asr/finetune_whisper.py "
        f"--train-jsonl {args.output} "
        f"--output-dir ./models/whisper-medium-pld-{normalize_pld_language(args.language)} "
        f"--model-name openai/whisper-medium --fp16"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
