#!/usr/bin/env python3
"""Evaluate stock mBART on Hiligaynon → English (isolated; no NLLB/Google/Whisper).

Scenarios the product sees when Whisper mislabels Ilonggo as Tagalog:

1. ``hil_clean`` + src ``hil`` → mBART ``id_ID`` (degraded native hil proxy)
2. ``hil_clean`` + src ``tl`` → mBART ``tl_XX`` (Hiligaynon mistagged as Tagalog)
3. ``hil_as_tl_noise`` + src ``tl`` → mBART ``tl_XX`` (ASR Tagalogized Hiligaynon)
4. ``tagalog`` + src ``tl`` → mBART ``tl_XX`` (Tagalog→English control)

Uses ``app.services.llm._mbart_translate`` only.

Example::

  python scripts/ph_mt/eval_mbart_hiligaynon.py \\
    --out data/mt_tag_benchmark/mbart_hiligaynon_eval.json
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


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def run_condition(
    rows: list[dict],
    *,
    src_code: str,
    label: str,
    skip_lexicon: bool = False,
) -> dict:
    from contextlib import ExitStack
    from unittest.mock import patch

    from app.services import llm

    detail = []
    t0 = time.perf_counter()
    with ExitStack() as stack:
        if skip_lexicon:
            stack.enter_context(
                patch("app.services.tagalog_phrases.lookup_exact", return_value=None)
            )
            try:
                from app.services import hiligaynon_phrases  # type: ignore

                stack.enter_context(
                    patch.object(hiligaynon_phrases, "lookup_exact", return_value=None)
                )
            except Exception:
                pass
        for row in rows:
            src = row.get("source") or row.get("src") or ""
            ref = row.get("reference") or row.get("tgt") or ""
            hyp = ""
            err = ""
            try:
                hyp = llm._mbart_translate(src, src_code, "en") or ""
            except llm._NonEnglishTranslation as exc:
                err = f"rejected:{exc}"
                hyp = ""
            except Exception as exc:  # noqa: BLE001 — eval harness
                err = f"{type(exc).__name__}:{exc}"
                hyp = ""
            f1 = token_f1(ref, hyp) if hyp else 0.0
            exact = tokenize(ref) == tokenize(hyp) if hyp else False
            detail.append(
                {
                    "src": src,
                    "ref": ref,
                    "hyp": hyp or err,
                    "token_f1": f1,
                    "exact": exact,
                    "note": row.get("note") or "",
                }
            )
            shown = (hyp or err)[:90]
            print(f"[{label}] F1={f1:.3f} | {shown}")

    mean = sum(d["token_f1"] for d in detail) / max(1, len(detail))
    exact_rate = sum(1 for d in detail if d["exact"]) / max(1, len(detail))
    return {
        "label": label,
        "src_code": src_code,
        "mean_token_f1": mean,
        "exact_match_rate": exact_rate,
        "seconds": round(time.perf_counter() - t0, 2),
        "n": len(detail),
        "rows": detail,
    }


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--hil-fixtures",
        type=Path,
        default=root / "scripts/ph_mt/fixtures/hiligaynon_en_sample.jsonl",
    )
    p.add_argument(
        "--hil-as-tl-fixtures",
        type=Path,
        default=root / "scripts/ph_mt/fixtures/hiligaynon_as_tagalog_noise.jsonl",
    )
    p.add_argument(
        "--tl-fixtures",
        type=Path,
        default=root / "scripts/ph_mt/fixtures/tagalog_en_sample.jsonl",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=root / "data/mt_tag_benchmark/mbart_hiligaynon_eval.json",
    )
    p.add_argument(
        "--neural-only",
        action="store_true",
        help="Disable exact phrase lexicons inside mBART path",
    )
    args = p.parse_args(argv)

    sys.path.insert(0, str(root / "backend"))

    hil = load_jsonl(args.hil_fixtures)
    hil_noise = load_jsonl(args.hil_as_tl_fixtures)
    tl = load_jsonl(args.tl_fixtures)
    if not hil or not hil_noise or not tl:
        print("Missing fixtures.", file=sys.stderr)
        return 1

    report: dict = {
        "scope": "mBART-only Hiligaynon / Tagalogized-Hiligaynon / Tagalog→EN",
        "neural_only": bool(args.neural_only),
        "conditions": {},
    }

    conditions = [
        (hil, "hil", "hil_clean→id_ID (hil proxy)"),
        (hil, "tl", "hil_clean→tl_XX (mistagged as Tagalog)"),
        (hil_noise, "tl", "hil_as_tl_noise→tl_XX"),
        (hil_noise, "hil", "hil_as_tl_noise→id_ID"),
        (tl, "tl", "tagalog→tl_XX control"),
    ]
    for rows, src, label in conditions:
        key = label.split()[0].replace("→", "_to_")
        print(f"\n== {label} ==")
        report["conditions"][key] = run_condition(
            rows,
            src_code=src,
            label=label,
            skip_lexicon=bool(args.neural_only),
        )
        c = report["conditions"][key]
        print(
            f"== {label}: mean F1={c['mean_token_f1']:.3f} "
            f"exact={c['exact_match_rate']:.0%} ({c['seconds']}s)"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
