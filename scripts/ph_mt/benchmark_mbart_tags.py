#!/usr/bin/env python3
"""Benchmark stock mBART language tags: tl_XX vs id_ID (and Hiligaynon checks).

Empirically validates the translation-tag choice called out in product review:

1. Tagalog: compare ``tl_XX`` vs current stock ``id_ID`` substitution on real
   Tagalog→English pairs (chrF++ / BLEU when ``sacrebleu`` is installed; always
   records model outputs for human review).
2. Hiligaynon: run the same stock mBART path via ``id_ID`` (least safety net)
   and write a human-eval worksheet — do **not** invent scores.

Usage::

  # Offline harness (no GPU) — uses fixtures + documents the protocol
  python3 scripts/ph_mt/benchmark_mbart_tags.py --fixtures-only \\
    --out-dir ./data/mt_tag_benchmark

  # Live model run (needs transformers + torch + MBART weights)
  python3 scripts/ph_mt/benchmark_mbart_tags.py \\
    --tagalog-jsonl scripts/ph_mt/fixtures/tagalog_en_sample.jsonl \\
    --hiligaynon-jsonl scripts/ph_mt/fixtures/hiligaynon_en_sample.jsonl \\
    --out-dir ./data/mt_tag_benchmark

See docs/MT_TAG_BENCHMARK.md.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tagalog-jsonl",
        type=Path,
        default=ROOT / "scripts/ph_mt/fixtures/tagalog_en_sample.jsonl",
    )
    p.add_argument(
        "--hiligaynon-jsonl",
        type=Path,
        default=ROOT / "scripts/ph_mt/fixtures/hiligaynon_en_sample.jsonl",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--model",
        default="facebook/mbart-large-50-many-to-many-mmt",
    )
    p.add_argument(
        "--fixtures-only",
        action="store_true",
        help="Skip model load; write protocol + fixture checklist only",
    )
    p.add_argument("--max-samples", type=int, default=50)
    return p.parse_args(argv)


def load_jsonl(path: Path, limit: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            src = (obj.get("source") or obj.get("src") or "").strip()
            ref = (obj.get("reference") or obj.get("tgt") or obj.get("en") or "").strip()
            if src and ref:
                rows.append({"source": src, "reference": ref})
            if len(rows) >= limit:
                break
    return rows


def _chrf(hyp: str, ref: str) -> float | None:
    try:
        from sacrebleu.metrics import CHRF

        return float(CHRF().sentence_score(hyp, [ref]).score)
    except Exception:
        return None


def _token_f1(hyp: str, ref: str) -> float:
    h = {t.lower() for t in hyp.split() if len(t) > 2}
    r = {t.lower() for t in ref.split() if len(t) > 2}
    if not h or not r:
        return 0.0
    inter = len(h & r)
    if inter == 0:
        return 0.0
    prec = inter / len(h)
    rec = inter / len(r)
    return 2 * prec * rec / (prec + rec)


def translate_batch(
    model_id: str, texts: list[str], src_code: str, tgt_code: str = "en_XX"
) -> list[str]:
    from transformers import MBart50TokenizerFast, MBartForConditionalGeneration

    tok = MBart50TokenizerFast.from_pretrained(model_id)
    model = MBartForConditionalGeneration.from_pretrained(model_id)
    tok.src_lang = src_code
    outs: list[str] = []
    for text in texts:
        encoded = tok(text, return_tensors="pt", truncation=True, max_length=256)
        generated = model.generate(
            **encoded,
            forced_bos_token_id=tok.lang_code_to_id[tgt_code],
            max_new_tokens=128,
            num_beams=4,
        )
        outs.append(tok.batch_decode(generated, skip_special_tokens=True)[0])
    return outs


def score_pair(hyp: str, ref: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "hypothesis": hyp,
        "reference": ref,
        "token_f1": round(_token_f1(hyp, ref), 4),
    }
    chrf = _chrf(hyp, ref)
    if chrf is not None:
        row["chrf"] = round(chrf, 4)
    return row


def summarize_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    f1s = [r["token_f1"] for r in rows]
    out: dict[str, Any] = {
        "n": len(rows),
        "token_f1_mean": round(statistics.mean(f1s), 4),
        "token_f1_median": round(statistics.median(f1s), 4),
    }
    chrfs = [r["chrf"] for r in rows if "chrf" in r]
    if chrfs:
        out["chrf_mean"] = round(statistics.mean(chrfs), 4)
    return out


def write_human_eval_sheet(path: Path, rows: list[dict[str, str]], tag: str) -> None:
    lines = [
        f"# Human eval worksheet — Hiligaynon via mBART `{tag}`",
        "",
        "Rate each item 1–5 for **adequacy** (meaning preserved) and **fluency**",
        "(natural English). Do not invent scores in automation — humans only.",
        "",
        "| # | Source (Hiligaynon) | Model English | Adequacy 1–5 | Fluency 1–5 | Notes |",
        "|---|---|---|---|---|---|",
    ]
    for i, row in enumerate(rows, 1):
        src = row["source"].replace("|", "/")
        hyp = (row.get("hypothesis") or "").replace("|", "/")
        lines.append(f"| {i} | {src} | {hyp} |  |  |  |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    tl_rows = load_jsonl(args.tagalog_jsonl, args.max_samples)
    hil_rows = load_jsonl(args.hiligaynon_jsonl, args.max_samples)

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "fixtures_only": bool(args.fixtures_only),
        "tagalog_n": len(tl_rows),
        "hiligaynon_n": len(hil_rows),
        "recommendation_pending": True,
        "notes": [
            "Stock Smart Meeting uses id_ID for hil/tl without a PH fine-tune.",
            "NLLB remains the default PH→EN path (tgl_Latn / ceb_Latn).",
            "This benchmark isolates stock mBART tag choice only.",
        ],
    }

    if args.fixtures_only or not tl_rows:
        report["status"] = "protocol_ready"
        report["next_steps"] = [
            "Run without --fixtures-only on a machine with transformers+torch.",
            "Compare tagalog.tl_XX.summary vs tagalog.id_ID.summary.",
            "Complete hiligaynon_id_ID_human_eval.md with 2+ bilingual raters.",
        ]
        (out / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        write_human_eval_sheet(
            out / "hiligaynon_id_ID_human_eval.md",
            [{**r, "hypothesis": "(run live model to fill)"} for r in hil_rows]
            or [
                {
                    "source": "(add Hiligaynon lines to fixtures)",
                    "hypothesis": "",
                }
            ],
            "id_ID",
        )
        print(json.dumps(report, indent=2))
        print(f"wrote {out / 'report.json'}")
        return 0

    # Live Tagalog: tl_XX vs id_ID
    sources = [r["source"] for r in tl_rows]
    refs = [r["reference"] for r in tl_rows]
    print("Translating Tagalog with tl_XX…", flush=True)
    hyp_tl = translate_batch(args.model, sources, "tl_XX")
    print("Translating Tagalog with id_ID…", flush=True)
    hyp_id = translate_batch(args.model, sources, "id_ID")

    scored_tl = [
        {"source": s, **score_pair(h, r)} for s, h, r in zip(sources, hyp_tl, refs)
    ]
    scored_id = [
        {"source": s, **score_pair(h, r)} for s, h, r in zip(sources, hyp_id, refs)
    ]
    report["tagalog"] = {
        "tl_XX": summarize_scores(scored_tl),
        "id_ID": summarize_scores(scored_id),
    }
    (out / "tagalog_tl_XX.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in scored_tl) + "\n",
        encoding="utf-8",
    )
    (out / "tagalog_id_ID.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in scored_id) + "\n",
        encoding="utf-8",
    )

    # Hiligaynon via id_ID — outputs for human eval (no automated winner).
    if hil_rows:
        print("Translating Hiligaynon with id_ID for human eval…", flush=True)
        hil_src = [r["source"] for r in hil_rows]
        hil_hyp = translate_batch(args.model, hil_src, "id_ID")
        hil_out = [
            {"source": s, "reference": r["reference"], "hypothesis": h}
            for s, r, h in zip(hil_src, hil_rows, hil_hyp)
        ]
        (out / "hiligaynon_id_ID.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in hil_out) + "\n",
            encoding="utf-8",
        )
        write_human_eval_sheet(out / "hiligaynon_id_ID_human_eval.md", hil_out, "id_ID")

    tl_mean = report["tagalog"]["tl_XX"].get("token_f1_mean", 0)
    id_mean = report["tagalog"]["id_ID"].get("token_f1_mean", 0)
    if tl_mean > id_mean + 0.02:
        report["tagalog_recommendation"] = (
            "Prefer tl_XX for stock mBART Tagalog (higher automatic overlap)."
        )
    elif id_mean > tl_mean + 0.02:
        report["tagalog_recommendation"] = (
            "id_ID scored higher on this sample — re-check with larger set "
            "before changing defaults; NLLB should remain primary PH→EN."
        )
    else:
        report["tagalog_recommendation"] = (
            "Scores similar on this sample — keep NLLB-first; treat mBART tag "
            "as secondary. Prefer tl_XX when a PH fine-tune is loaded."
        )
    report["recommendation_pending"] = False
    report["status"] = "complete"
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
