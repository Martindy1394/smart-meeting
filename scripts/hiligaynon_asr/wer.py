#!/usr/bin/env python3
"""Simple Word Error Rate helpers for Hiligaynon ASR evaluation.

Usage:
  python scripts/hiligaynon_asr/wer.py --reference ref.txt --hypothesis hyp.txt
  python scripts/hiligaynon_asr/wer.py --pair clips/eval.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ0-9']+")


def normalize_text(text: str) -> str:
    return " ".join(_TOKEN_RE.findall((text or "").lower()))


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    """Levenshtein distance over word tokens."""
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i]
        for j, h in enumerate(hyp, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if r == h else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def word_error_rate(reference: str, hypothesis: str) -> dict:
    ref = tokenize(reference)
    hyp = tokenize(hypothesis)
    dist = edit_distance(ref, hyp)
    n = max(1, len(ref))
    return {
        "wer": dist / n,
        "wer_percent": 100.0 * dist / n,
        "edits": dist,
        "ref_words": len(ref),
        "hyp_words": len(hyp),
        "reference_norm": " ".join(ref),
        "hypothesis_norm": " ".join(hyp),
    }


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, help="Reference transcript .txt")
    p.add_argument("--hypothesis", type=Path, help="ASR hypothesis .txt")
    p.add_argument(
        "--pair",
        type=Path,
        help="JSONL with {reference, hypothesis} or {text, prediction} rows",
    )
    args = p.parse_args(argv)

    rows: list[tuple[str, str]] = []
    if args.pair:
        for line in args.pair.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            ref = obj.get("reference") or obj.get("text") or ""
            hyp = obj.get("hypothesis") or obj.get("prediction") or ""
            rows.append((ref, hyp))
    elif args.reference and args.hypothesis:
        rows.append((_load_text(args.reference), _load_text(args.hypothesis)))
    else:
        p.error("Provide --reference/--hypothesis or --pair")

    total_edits = 0
    total_ref = 0
    for i, (ref, hyp) in enumerate(rows, start=1):
        m = word_error_rate(ref, hyp)
        total_edits += m["edits"]
        total_ref += m["ref_words"]
        print(
            f"[{i}] WER={m['wer_percent']:.2f}% "
            f"(edits={m['edits']} ref_words={m['ref_words']})"
        )

    overall = total_edits / max(1, total_ref)
    print(f"overall_wer_percent={100.0 * overall:.2f}")
    print(f"overall_edits={total_edits} overall_ref_words={total_ref}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
