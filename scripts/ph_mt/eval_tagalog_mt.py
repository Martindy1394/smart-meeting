#!/usr/bin/env python3
"""Evaluate Tagalog→English MT (phrase lexicon / NLLB / mBART) on fixtures.

Reports mean token-F1 vs references. Does **not** claim 100% neural accuracy —
only curated phrase-lexicon hits are exact.

Example:
  python scripts/ph_mt/eval_tagalog_mt.py \\
    --fixtures scripts/ph_mt/fixtures/tagalog_en_sample.jsonl \\
    --engines lexicon,nllb,mbart
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ0-9']+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def token_f1(ref: str, hyp: str) -> float:
    r, h = set(tokenize(ref)), set(tokenize(hyp))
    if not r and not h:
        return 1.0
    if not r or not h:
        return 0.0
    inter = len(r & h)
    p, rec = inter / len(h), inter / len(r)
    return 0.0 if p + rec == 0 else 2 * p * rec / (p + rec)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--fixtures",
        type=Path,
        default=Path("scripts/ph_mt/fixtures/tagalog_en_sample.jsonl"),
    )
    p.add_argument(
        "--engines",
        default="lexicon,nllb,mbart",
        help="Comma list: lexicon,nllb,mbart,pipeline",
    )
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    rows = []
    for line in args.fixtures.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        print("No fixture rows.", file=sys.stderr)
        return 1

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "backend"))

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    report: dict = {"fixtures": str(args.fixtures), "engines": {}}

    for eng in engines:
        scores = []
        detail = []
        t0 = time.perf_counter()
        for row in rows:
            src = row.get("source") or row.get("src") or ""
            ref = row.get("reference") or row.get("tgt") or ""
            hyp = ""
            used = eng
            if eng == "lexicon":
                from app.services import tagalog_phrases

                hyp = tagalog_phrases.lookup_exact(src) or ""
                used = "tagalog-phrase-lexicon" if hyp else "miss"
            elif eng == "nllb":
                from app.services import llm

                hyp = llm._nllb_translate_to_english(src, "tl")
            elif eng == "mbart":
                from app.services import llm

                hyp = llm._mbart_translate(src, "tl", "en")
            elif eng == "pipeline":
                from app.services import llm

                tr = llm.translate(src, target_language="en", source_language="tl")
                hyp = tr.text or ""
                used = tr.engine
            else:
                print(f"Unknown engine {eng}", file=sys.stderr)
                return 2
            f1 = token_f1(ref, hyp) if hyp else 0.0
            exact = tokenize(ref) == tokenize(hyp) if hyp else False
            scores.append(f1)
            detail.append(
                {
                    "src": src,
                    "ref": ref,
                    "hyp": hyp,
                    "token_f1": f1,
                    "exact": exact,
                    "engine": used,
                }
            )
            print(f"[{eng}] F1={f1:.3f} exact={exact} | {hyp[:80]}")
        mean = sum(scores) / len(scores)
        exact_rate = sum(1 for d in detail if d["exact"]) / len(detail)
        report["engines"][eng] = {
            "mean_token_f1": mean,
            "exact_match_rate": exact_rate,
            "seconds": round(time.perf_counter() - t0, 2),
            "rows": detail,
        }
        print(
            f"== {eng}: mean F1={mean:.3f} exact={exact_rate:.0%} "
            f"({time.perf_counter()-t0:.1f}s)"
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("Wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
