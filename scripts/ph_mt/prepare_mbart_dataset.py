#!/usr/bin/env python3
"""Build mBART fine-tune JSONL for Tagalog / Hiligaynon → English.

Sources (in priority for quality):
  1. OPUS Tatoeba English–Tagalog Moses pair files
  2. Curated Hiligaynon meeting seed JSONL
  3. Optional pinoy-dictionary-scraper gloss pairs (capped — weak signal)

Output rows:
  {"src": "...", "tgt": "...", "src_lang": "tl_XX", "tgt_lang": "en_XX", "language": "tl"|"hil"}

Example:
  python scripts/ph_mt/prepare_mbart_dataset.py \\
    --tatoeba-dir /tmp/opus-tl \\
    --dictionary-dir scripts/ph_mt/data \\
    --hil-seed scripts/ph_mt/seed_hiligaynon_en.jsonl \\
    --output-dir scripts/ph_mt/prepared
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

_POS_RE = re.compile(
    r"^\s*(?:n\.|v\.|adj\.|adv\.|prep\.|conj\.|pron\.|interj\.|a\.|vt\.|vi\.)\s*",
    re.I,
)
_WORD_PREFIX_RE = re.compile(r"^[A-Za-zÀ-ÿ\-']+\s+(?:adj\.|n\.|v\.|adv\.)\s*", re.I)


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_tatoeba(en_path: Path, tl_path: Path) -> list[dict]:
    en_lines = en_path.read_text(encoding="utf-8").splitlines()
    tl_lines = tl_path.read_text(encoding="utf-8").splitlines()
    n = min(len(en_lines), len(tl_lines))
    out: list[dict] = []
    for i in range(n):
        src = _clean(tl_lines[i])
        tgt = _clean(en_lines[i])
        if not src or not tgt:
            continue
        if src.lower() == tgt.lower():
            continue
        out.append(
            {
                "src": src,
                "tgt": tgt,
                "src_lang": "tl_XX",
                "tgt_lang": "en_XX",
                "language": "tl",
            }
        )
    return out


def _gloss_from_definition(word: str, definition: str) -> str | None:
    """Pull a short English gloss from a dictionary definition when possible."""
    text = _clean(definition)
    if not text:
        return None
    # Drop leading "word pos." echoes.
    text = _WORD_PREFIX_RE.sub("", text)
    if text.lower().startswith(word.lower()):
        text = text[len(word) :].lstrip(" .-:")
    text = _POS_RE.sub("", text)
    # Keep first short clause.
    for sep in (";", ".", "—", " - "):
        if sep in text:
            text = text.split(sep, 1)[0]
    text = _clean(text)
    if not text or len(text) < 2 or len(text) > 80:
        return None
    # Skip long grammar notes.
    if len(text.split()) > 12:
        return None
    if text.lower() == word.lower():
        return None
    return text


def load_dictionary_pairs(path: Path, language: str, limit: int) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return []
    pairs: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        word = _clean(str(item.get("word") or ""))
        definition = str(item.get("definition") or "")
        if not word or len(word) < 2 or " " in word and len(word.split()) > 3:
            continue
        gloss = _gloss_from_definition(word, definition)
        if not gloss:
            continue
        pairs.append(
            {
                "src": word,
                "tgt": gloss,
                "src_lang": "tl_XX",
                "tgt_lang": "en_XX",
                "language": language,
                "source": "pinoy-dictionary",
            }
        )
    random.shuffle(pairs)
    return pairs[: max(0, limit)]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tatoeba-dir", type=Path, default=None)
    p.add_argument("--dictionary-dir", type=Path, default=None)
    p.add_argument(
        "--hil-seed",
        type=Path,
        default=Path("scripts/ph_mt/seed_hiligaynon_en.jsonl"),
    )
    p.add_argument(
        "--tl-seed",
        type=Path,
        default=Path("scripts/ph_mt/seed_tagalog_en.jsonl"),
        help="Curated Tagalog meeting-domain seed JSONL",
    )
    p.add_argument(
        "--lang",
        choices=("all", "tl", "hil"),
        default="all",
        help="Keep only Tagalog, only Hiligaynon proxy rows, or both",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dict-limit-per-lang", type=int, default=800)
    p.add_argument("--eval-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=13)
    args = p.parse_args(argv)

    random.seed(args.seed)
    rows: list[dict] = []

    if args.tatoeba_dir:
        en = args.tatoeba_dir / "Tatoeba.en-tl.en"
        tl = args.tatoeba_dir / "Tatoeba.en-tl.tl"
        if en.exists() and tl.exists():
            tl_rows = load_tatoeba(en, tl)
            rows.extend(tl_rows)
            print(f"Tatoeba TL-EN: {len(tl_rows)}")
        else:
            print(f"Tatoeba files missing under {args.tatoeba_dir}", file=sys.stderr)

    if args.tl_seed and args.tl_seed.exists():
        tl_seed = _load_jsonl(args.tl_seed)
        rows.extend(tl_seed)
        print(f"Tagalog meeting seed: {len(tl_seed)}")

    if args.hil_seed and args.hil_seed.exists() and args.lang in {"all", "hil"}:
        hil = _load_jsonl(args.hil_seed)
        rows.extend(hil)
        print(f"Hiligaynon seed: {len(hil)}")

    if args.dictionary_dir:
        mapping = {
            "tagalog_dictionary.json": "tl",
            "hiligaynon_dictionary.json": "hil",
            "filipino_dictionary.json": "tl",
        }
        for name, lang in mapping.items():
            path = args.dictionary_dir / name
            if not path.exists():
                continue
            dict_rows = load_dictionary_pairs(path, lang, args.dict_limit_per_lang)
            rows.extend(dict_rows)
            print(f"Dictionary {name}: {len(dict_rows)} gloss pairs")

    # Deduplicate by src|tgt
    seen: set[str] = set()
    unique: list[dict] = []
    for row in rows:
        lang = (row.get("language") or "tl").strip().lower()
        if args.lang == "tl" and lang not in {"tl", "tagalog", "fil", "filipino"}:
            continue
        if args.lang == "hil" and lang not in {"hil", "hiligaynon", "ilonggo"}:
            continue
        key = f"{row['src'].lower()}||{row['tgt'].lower()}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)

    if not unique:
        print("No training rows produced.", file=sys.stderr)
        return 1

    random.shuffle(unique)
    n_eval = max(1, int(len(unique) * args.eval_ratio))
    eval_rows = unique[:n_eval]
    train_rows = unique[n_eval:]

    out = args.output_dir
    write_jsonl(out / "train.jsonl", train_rows)
    write_jsonl(out / "eval.jsonl", eval_rows)
    meta = {
        "train": len(train_rows),
        "eval": len(eval_rows),
        "by_language": {
            "tl": sum(1 for r in unique if r.get("language") == "tl"),
            "hil": sum(1 for r in unique if r.get("language") == "hil"),
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"Wrote {out / 'train.jsonl'} and {out / 'eval.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
