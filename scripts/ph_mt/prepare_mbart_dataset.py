#!/usr/bin/env python3
"""Build mBART fine-tune JSONL for Tagalog or Hiligaynon → English.

Two different jobs (do not mix casually):

  --lang tl   native ``tl_XX`` (no vocab change)
  --lang hil  rows tagged ``hil_XX`` (tokenizer extended at train time)

Sources
  Tagalog: FLORES-200 ``tgl_Latn`` (held-out by default), OPUS Tatoeba /
  OPUS-100, domain meeting/lyric seeds, optional pinoy-dictionary glosses.
  Hiligaynon: curated meeting seed, optional dictionary glosses, optional
  OPUS / SEACrowd paths when provided locally (data is scarce).

Example:
  python scripts/ph_mt/prepare_mbart_dataset.py --lang tl \\
    --download-opus --download-flores \\
    --domain-seed scripts/ph_mt/seed_tagalog_en.jsonl \\
    --output-dir scripts/ph_mt/prepared/tl

  python scripts/ph_mt/prepare_mbart_dataset.py --lang hil \\
    --hil-seed scripts/ph_mt/seed_hiligaynon_en.jsonl \\
    --output-dir scripts/ph_mt/prepared/hil
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

_POS_RE = re.compile(
    r"^\s*(?:n\.|v\.|adj\.|adv\.|prep\.|conj\.|pron\.|interj\.|a\.|vt\.|vi\.)\s*",
    re.I,
)
_WORD_PREFIX_RE = re.compile(r"^[A-Za-zÀ-ÿ\-']+\s+(?:adj\.|n\.|v\.|adv\.)\s*", re.I)

TATOEBA_URL = (
    "https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/en-tl.txt.zip"
)


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _row(
    src: str,
    tgt: str,
    *,
    language: str,
    src_lang: str,
    source: str,
) -> dict | None:
    src, tgt = _clean(src), _clean(tgt)
    if not src or not tgt or src.lower() == tgt.lower():
        return None
    return {
        "src": src,
        "tgt": tgt,
        "src_lang": src_lang,
        "tgt_lang": "en_XX",
        "language": language,
        "source": source,
    }


def load_tatoeba(en_path: Path, tl_path: Path, src_lang: str) -> list[dict]:
    en_lines = en_path.read_text(encoding="utf-8").splitlines()
    tl_lines = tl_path.read_text(encoding="utf-8").splitlines()
    n = min(len(en_lines), len(tl_lines))
    out: list[dict] = []
    for i in range(n):
        row = _row(
            tl_lines[i],
            en_lines[i],
            language="tl",
            src_lang=src_lang,
            source="opus-tatoeba",
        )
        if row:
            out.append(row)
    return out


def download_tatoeba(dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "en-tl.txt.zip"
    if not (dest_dir / "Tatoeba.en-tl.en").exists():
        print(f"Downloading Tatoeba en-tl → {zip_path}")
        urllib.request.urlretrieve(TATOEBA_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
    return dest_dir


def load_opus100(src_lang: str, max_rows: int) -> list[dict]:
    """Optional broader OPUS-100 en-tl (JW300/GV/etc. are OPUS-family)."""
    try:
        from datasets import load_dataset
    except Exception as exc:
        print(f"datasets unavailable for OPUS-100: {exc}", file=sys.stderr)
        return []
    print(f"Loading Helsinki-NLP/opus-100 en-tl (max={max_rows}) …")
    try:
        ds = load_dataset("Helsinki-NLP/opus-100", "en-tl", split="train")
    except Exception as exc:
        print(f"OPUS-100 load failed: {exc}", file=sys.stderr)
        return []
    out: list[dict] = []
    for i, ex in enumerate(ds):
        if i >= max_rows:
            break
        # opus-100 rows: {"translation": {"en": ..., "tl": ...}}
        tr = ex.get("translation") or ex
        src = tr.get("tl") or tr.get("tgl") or ""
        tgt = tr.get("en") or ""
        row = _row(src, tgt, language="tl", src_lang=src_lang, source="opus-100")
        if row:
            out.append(row)
    print(f"OPUS-100 rows kept: {len(out)}")
    return out


def load_flores_tgl(src_lang: str, split: str = "devtest") -> list[dict]:
    """FLORES-200 Tagalog↔English (prefer held-out / eval, not train)."""
    try:
        from datasets import load_dataset
    except Exception as exc:
        print(f"datasets unavailable for FLORES: {exc}", file=sys.stderr)
        return []
    print(f"Loading facebook/flores tgl_Latn ({split}) …")
    # Try a few known config layouts.
    attempts = [
        ("facebook/flores", "tgl_Latn", split),
        ("facebook/flores", "eng_Latn-tgl_Latn", split),
        ("facebook/flores-200", "tgl_Latn", split),
    ]
    ds = None
    last_err: Exception | None = None
    for name, config, sp in attempts:
        try:
            ds = load_dataset(name, config, split=sp, trust_remote_code=True)
            print(f"  loaded {name}/{config}")
            break
        except Exception as exc:
            last_err = exc
            continue
    if ds is None:
        print(f"FLORES load failed: {last_err}", file=sys.stderr)
        return []

    out: list[dict] = []
    for ex in ds:
        # Common key patterns across FLORES HF mirrors.
        src = (
            ex.get("sentence_tgl_Latn")
            or ex.get("tgl_Latn")
            or (ex.get("translation") or {}).get("tgl_Latn")
            or ""
        )
        tgt = (
            ex.get("sentence_eng_Latn")
            or ex.get("eng_Latn")
            or (ex.get("translation") or {}).get("eng_Latn")
            or ""
        )
        row = _row(src, tgt, language="tl", src_lang=src_lang, source=f"flores-{split}")
        if row:
            out.append(row)
    print(f"FLORES rows: {len(out)}")
    return out


def load_domain_seed(path: Path, language: str, src_lang: str) -> list[dict]:
    if not path or not path.exists():
        return []
    out: list[dict] = []
    for raw in _load_jsonl(path):
        src = raw.get("src") or raw.get("source") or ""
        tgt = raw.get("tgt") or raw.get("reference") or raw.get("en") or ""
        row = _row(
            src,
            tgt,
            language=language,
            src_lang=src_lang,
            source=str(raw.get("source") or "domain-seed"),
        )
        if row:
            out.append(row)
    return out


def load_fixture_holdout(path: Path, language: str, src_lang: str) -> list[dict]:
    """Meeting-fixture JSONL always reserved for domain eval."""
    if not path.exists():
        return []
    out: list[dict] = []
    for raw in _load_jsonl(path):
        src = raw.get("source") or raw.get("src") or ""
        tgt = raw.get("reference") or raw.get("tgt") or ""
        row = _row(
            src,
            tgt,
            language=language,
            src_lang=src_lang,
            source="domain-fixture",
        )
        if row:
            out.append(row)
    return out


def _gloss_from_definition(word: str, definition: str) -> str | None:
    text = _clean(definition)
    if not text:
        return None
    text = _WORD_PREFIX_RE.sub("", text)
    if text.lower().startswith(word.lower()):
        text = text[len(word) :].lstrip(" .-:")
    text = _POS_RE.sub("", text)
    for sep in (";", ".", "—", " - "):
        if sep in text:
            text = text.split(sep, 1)[0]
    text = _clean(text)
    if not text or len(text) < 2 or len(text) > 80:
        return None
    if len(text.split()) > 12 or text.lower() == word.lower():
        return None
    return text


def load_dictionary_pairs(
    path: Path, language: str, src_lang: str, limit: int
) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return []
    pairs: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        word = _clean(str(item.get("word") or ""))
        definition = str(item.get("definition") or "")
        if not word or len(word) < 2 or (" " in word and len(word.split()) > 3):
            continue
        gloss = _gloss_from_definition(word, definition)
        if not gloss:
            continue
        pairs.append(
            {
                "src": word,
                "tgt": gloss,
                "src_lang": src_lang,
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


def _dedupe(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []
    for row in rows:
        key = f"{row['src'].lower()}||{row['tgt'].lower()}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--lang",
        choices=("tl", "hil"),
        required=True,
        help="tl = Tagalog/tl_XX; hil = Hiligaynon/hil_XX (vocab added at train)",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--tatoeba-dir", type=Path, default=None)
    p.add_argument("--download-opus", action="store_true", help="Fetch Tatoeba en-tl")
    p.add_argument(
        "--download-flores",
        action="store_true",
        help="Fetch FLORES-200 tgl (held out of train by default)",
    )
    p.add_argument(
        "--opus100-max",
        type=int,
        default=0,
        help="Also pull up to N OPUS-100 en-tl rows (0=skip)",
    )
    p.add_argument(
        "--flores-in-train",
        action="store_true",
        help="Allow FLORES into train (default: eval/domain only)",
    )
    p.add_argument("--dictionary-dir", type=Path, default=None)
    p.add_argument(
        "--domain-seed",
        type=Path,
        default=None,
        help="Domain meeting/lyric JSONL (defaults by --lang)",
    )
    p.add_argument(
        "--hil-seed",
        type=Path,
        default=Path("scripts/ph_mt/seed_hiligaynon_en.jsonl"),
    )
    p.add_argument(
        "--extra-jsonl",
        type=Path,
        nargs="*",
        default=[],
        help="Extra parallel JSONL (SEACrowd/OPUS Bible exports, etc.)",
    )
    p.add_argument("--dict-limit-per-lang", type=int, default=400)
    p.add_argument("--eval-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=13)
    args = p.parse_args(argv)

    random.seed(args.seed)
    language = args.lang
    src_lang = "tl_XX" if language == "tl" else "hil_XX"

    if args.domain_seed is None:
        args.domain_seed = (
            Path("scripts/ph_mt/seed_tagalog_en.jsonl")
            if language == "tl"
            else args.hil_seed
        )

    train_pool: list[dict] = []
    eval_pool: list[dict] = []

    # --- Tagalog web bitext ---
    if language == "tl":
        tatoeba_dir = args.tatoeba_dir
        if args.download_opus:
            tatoeba_dir = download_tatoeba(Path("/tmp/opus-tl"))
        if tatoeba_dir:
            en = tatoeba_dir / "Tatoeba.en-tl.en"
            tl = tatoeba_dir / "Tatoeba.en-tl.tl"
            if en.exists() and tl.exists():
                rows = load_tatoeba(en, tl, src_lang)
                train_pool.extend(rows)
                print(f"Tatoeba TL-EN: {len(rows)}")
            else:
                print(f"Tatoeba files missing under {tatoeba_dir}", file=sys.stderr)

        if args.opus100_max > 0:
            train_pool.extend(load_opus100(src_lang, args.opus100_max))

        if args.download_flores:
            flores = load_flores_tgl(src_lang, split="devtest")
            if args.flores_in_train:
                train_pool.extend(flores)
            else:
                eval_pool.extend(flores)
                print(f"FLORES held out for eval: {len(flores)}")

    # --- Domain seeds ---
    domain = load_domain_seed(args.domain_seed, language, src_lang)
    train_pool.extend(domain)
    print(f"Domain seed ({args.domain_seed}): {len(domain)}")

    if language == "hil" and args.hil_seed and args.hil_seed != args.domain_seed:
        if args.hil_seed.exists():
            hil = load_domain_seed(args.hil_seed, "hil", src_lang)
            train_pool.extend(hil)
            print(f"Hiligaynon seed: {len(hil)}")

    for extra in args.extra_jsonl or []:
        if extra.exists():
            rows = load_domain_seed(extra, language, src_lang)
            train_pool.extend(rows)
            print(f"Extra {extra}: {len(rows)}")

    # --- Optional dictionaries (weak) ---
    if args.dictionary_dir:
        names = (
            ["tagalog_dictionary.json", "filipino_dictionary.json"]
            if language == "tl"
            else ["hiligaynon_dictionary.json"]
        )
        for name in names:
            path = args.dictionary_dir / name
            if not path.exists():
                continue
            dict_rows = load_dictionary_pairs(
                path, language, src_lang, args.dict_limit_per_lang
            )
            train_pool.extend(dict_rows)
            print(f"Dictionary {name}: {len(dict_rows)}")

    # --- Meeting fixtures always domain-eval (never train) ---
    fixture = Path(
        "scripts/ph_mt/fixtures/tagalog_en_sample.jsonl"
        if language == "tl"
        else "scripts/ph_mt/fixtures/hiligaynon_en_sample.jsonl"
    )
    fixture_rows = load_fixture_holdout(fixture, language, src_lang)
    eval_pool.extend(fixture_rows)
    print(f"Domain fixtures held out: {len(fixture_rows)}")

    # Remove any train rows that duplicate fixture sources
    fixture_src = {r["src"].lower() for r in fixture_rows}
    train_pool = [r for r in train_pool if r["src"].lower() not in fixture_src]

    train_pool = _dedupe(train_pool)
    eval_pool = _dedupe(eval_pool)

    if not train_pool:
        print("No training rows produced.", file=sys.stderr)
        return 1

    random.shuffle(train_pool)
    # Split a random held-out slice from train bitext (in addition to fixtures/FLORES)
    n_hold = max(1, int(len(train_pool) * args.eval_ratio)) if len(train_pool) > 20 else 0
    random_eval = train_pool[:n_hold] if n_hold else []
    train_rows = train_pool[n_hold:] if n_hold else train_pool
    eval_rows = _dedupe(eval_pool + random_eval)

    out = args.output_dir
    write_jsonl(out / "train.jsonl", train_rows)
    write_jsonl(out / "eval.jsonl", eval_rows)
    # Explicit domain-only eval file for go/no-go
    domain_eval = [r for r in eval_rows if str(r.get("source", "")).startswith("domain")]
    if not domain_eval:
        domain_eval = fixture_rows
    write_jsonl(out / "eval_domain.jsonl", domain_eval)

    meta = {
        "lang": language,
        "src_lang": src_lang,
        "tgt_lang": "en_XX",
        "train": len(train_rows),
        "eval": len(eval_rows),
        "eval_domain": len(domain_eval),
        "by_source": {},
    }
    for r in train_rows + eval_rows:
        src = str(r.get("source") or "unknown")
        meta["by_source"][src] = meta["by_source"].get(src, 0) + 1
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"Wrote {out / 'train.jsonl'}, {out / 'eval.jsonl'}, {out / 'eval_domain.jsonl'}")
    if language == "hil" and len(train_rows) < 200:
        print(
            "NOTE: Hiligaynon bitext is scarce — expect weaker scores than Tagalog; "
            "keep Google Translate as production primary.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
