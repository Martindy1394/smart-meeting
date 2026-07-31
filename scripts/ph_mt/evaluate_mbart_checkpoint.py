#!/usr/bin/env python3
"""Evaluate an mBART PH checkpoint vs stock id_ID / tl_XX baselines.

Scores BLEU, chrF, and token-F1 on:
  - held-out parallel eval JSONL (generic)
  - domain meeting/lyric fixtures (required for go/no-go)

Go criterion (Tagalog): domain token-F1 must clearly beat stock ``tl_XX``
baseline (~0.43 on meeting fixtures) and ``id_ID`` mistag (~0.34).

Example:
  python scripts/ph_mt/evaluate_mbart_checkpoint.py \\
    --checkpoint models/mbart-tl-en-merged \\
    --lang tl \\
    --domain-jsonl scripts/ph_mt/fixtures/tagalog_en_sample.jsonl \\
    --eval-jsonl scripts/ph_mt/prepared/tl/eval.jsonl \\
    --out data/mbart_ph_ft/tl_eval_report.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        src = obj.get("src") or obj.get("source") or ""
        tgt = obj.get("tgt") or obj.get("reference") or obj.get("en") or ""
        if src and tgt:
            rows.append({"src": src, "tgt": tgt, "source": obj.get("source")})
    return rows


def token_f1(hyp: str, ref: str) -> float:
    def toks(s: str) -> list[str]:
        return [t for t in re.findall(r"[a-z0-9']+", (s or "").lower()) if t]

    h, r = toks(hyp), toks(ref)
    if not h or not r:
        return 0.0
    hc, rc = Counter(h), Counter(r)
    overlap = sum(min(hc[t], rc[t]) for t in hc)
    prec = overlap / len(h)
    rec = overlap / len(r)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def corpus_bleu_chrf(hyps: list[str], refs: list[str]) -> tuple[float, float]:
    try:
        import sacrebleu
    except Exception:
        return -1.0, -1.0
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
    return float(bleu), float(chrf)


def translate_batch(
    model,
    tokenizer,
    texts: list[str],
    src_lang: str,
    *,
    max_new_tokens: int = 128,
) -> list[str]:
    import torch

    tokenizer.src_lang = src_lang
    bos = tokenizer.lang_code_to_id.get("en_XX")
    if bos is None:
        bos = tokenizer.convert_tokens_to_ids("en_XX")
    outs: list[str] = []
    for text in texts:
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.inference_mode():
            gen = model.generate(
                **enc,
                forced_bos_token_id=bos,
                max_new_tokens=min(
                    max_new_tokens, max(64, int(len(text.split()) * 2.8))
                ),
                num_beams=6,
                early_stopping=True,
                no_repeat_ngram_size=4,
                length_penalty=1.05,
            )
        outs.append(tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip())
    return outs


def score_set(hyps: list[str], refs: list[str]) -> dict:
    f1s = [token_f1(h, r) for h, r in zip(hyps, refs)]
    bleu, chrf = corpus_bleu_chrf(hyps, refs)
    return {
        "n": len(refs),
        "token_f1_mean": sum(f1s) / len(f1s) if f1s else 0.0,
        "token_f1_median": sorted(f1s)[len(f1s) // 2] if f1s else 0.0,
        "bleu": bleu,
        "chrf": chrf,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--lang", choices=("tl", "hil"), required=True)
    p.add_argument("--domain-jsonl", type=Path, required=True)
    p.add_argument("--eval-jsonl", type=Path, default=None)
    p.add_argument(
        "--base-model",
        default="facebook/mbart-large-50-many-to-many-mmt",
        help="Stock baseline model for id_ID / tl_XX comparison",
    )
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Only score the fine-tuned checkpoint",
    )
    args = p.parse_args(argv)

    try:
        from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
    except Exception as exc:
        print(f"transformers required: {exc}", file=sys.stderr)
        return 2

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _hil_xx_tokenizer import load_mbart_tokenizer

    domain = _load_jsonl(args.domain_jsonl)
    if not domain:
        print("Domain JSONL empty.", file=sys.stderr)
        return 1
    generic = _load_jsonl(args.eval_jsonl) if args.eval_jsonl and args.eval_jsonl.exists() else []

    print(f"Loading fine-tuned checkpoint {args.checkpoint}")
    ckpt = Path(args.checkpoint)
    ft_tok = load_mbart_tokenizer(ckpt) if ckpt.is_dir() else MBart50TokenizerFast.from_pretrained(str(args.checkpoint))
    ft_model = MBartForConditionalGeneration.from_pretrained(str(ckpt.resolve() if ckpt.is_dir() else args.checkpoint))
    ft_model.eval()

    src_ft = "tl_XX" if args.lang == "tl" else "hil_XX"
    if args.lang == "hil" and "hil_XX" not in ft_tok.lang_code_to_id:
        print(
            "WARNING: checkpoint has no hil_XX — falling back to tl_XX tag "
            "(legacy proxy). Retrain with --lang hil.",
            file=sys.stderr,
        )
        src_ft = "tl_XX"

    domain_srcs = [r["src"] for r in domain]
    domain_refs = [r["tgt"] for r in domain]

    print(f"Scoring fine-tune src_lang={src_ft} on {len(domain)} domain rows …")
    ft_domain_hyps = translate_batch(ft_model, ft_tok, domain_srcs, src_ft)
    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "lang": args.lang,
        "src_lang_ft": src_ft,
        "has_hil_xx": "hil_XX" in getattr(ft_tok, "lang_code_to_id", {}),
        "domain": {
            "fine_tune": score_set(ft_domain_hyps, domain_refs),
            "examples": [
                {"src": s, "ref": r, "hyp": h}
                for s, r, h in list(zip(domain_srcs, domain_refs, ft_domain_hyps))[:5]
            ],
        },
        "baselines": {},
        "generic_eval": {},
        "recommendation": {},
    }

    if generic:
        g_srcs = [r["src"] for r in generic]
        g_refs = [r["tgt"] for r in generic]
        print(f"Scoring fine-tune on {len(generic)} generic eval rows …")
        g_hyps = translate_batch(ft_model, ft_tok, g_srcs, src_ft)
        report["generic_eval"]["fine_tune"] = score_set(g_hyps, g_refs)

    # Free FT before loading baseline if memory is tight
    del ft_model

    if not args.skip_baselines:
        print(f"Loading stock baseline {args.base_model}")
        base_tok = MBart50TokenizerFast.from_pretrained(args.base_model)
        base_model = MBartForConditionalGeneration.from_pretrained(args.base_model)
        base_model.eval()
        for tag in ("tl_XX", "id_ID"):
            print(f"  baseline tag {tag} …")
            hyps = translate_batch(base_model, base_tok, domain_srcs, tag)
            report["baselines"][tag] = score_set(hyps, domain_refs)
            report["baselines"][tag]["examples"] = [
                {"src": s, "ref": r, "hyp": h}
                for s, r, h in list(zip(domain_srcs, domain_refs, hyps))[:3]
            ]

    # Go / no-go
    ft_f1 = report["domain"]["fine_tune"]["token_f1_mean"]
    tl_f1 = report["baselines"].get("tl_XX", {}).get("token_f1_mean")
    id_f1 = report["baselines"].get("id_ID", {}).get("token_f1_mean")
    # Historical fixture means from docs/MBART_PH_AUDIT.md when baselines skipped
    tl_ref = 0.43 if tl_f1 is None else tl_f1
    id_ref = 0.34 if id_f1 is None else id_f1

    ft_bleu = report["domain"]["fine_tune"].get("bleu") or -1
    ft_chrf = report["domain"]["fine_tune"].get("chrf") or -1
    if args.lang == "tl":
        beats_id = ft_f1 > id_ref + 0.02
        beats_tl = ft_f1 > tl_ref + 0.02
        metric_ok = beats_id and beats_tl
        # Absolute floors so tiny smoke runs that barely beat a weak baseline
        # do not auto-wire into production.
        quality_ok = ft_f1 >= 0.55 and (ft_bleu < 0 or ft_bleu >= 20.0)
        go = metric_ok and quality_ok
        reason = (
            f"domain token-F1={ft_f1:.3f} bleu={ft_bleu:.1f} chrf={ft_chrf:.1f}; "
            f"need > tl_XX({tl_ref:.3f})+0.02, > id_ID({id_ref:.3f})+0.02, "
            f"and floors F1≥0.55 / BLEU≥20"
        )
    else:
        beats_id = ft_f1 > id_ref + 0.05
        beats_tl = tl_f1 is None or ft_f1 > tl_f1 + 0.02
        metric_ok = beats_id and beats_tl and report.get("has_hil_xx", False)
        quality_ok = ft_f1 >= 0.40 and (ft_bleu < 0 or ft_bleu >= 12.0)
        go = metric_ok and quality_ok
        reason = (
            f"domain token-F1={ft_f1:.3f} bleu={ft_bleu:.1f}; need hil_XX in vocab, "
            f"> id_ID({id_ref:.3f})+0.05, and floors F1≥0.40 / BLEU≥12"
        )

    report["recommendation"] = {
        "go": go,
        "decision": "GO" if go else ("PROVISIONAL" if metric_ok else "NO-GO"),
        "metric_improvement": metric_ok,
        "quality_floor_met": quality_ok,
        "reason": reason,
        "wire_MBART_PH_FINE_TUNED_MODEL": go,
        "notes": [
            "PROVISIONAL = beats stock tags on domain F1 but below absolute "
            "quality floors — retrain on GPU (≥1–3 epochs) before deploy.",
            "Hiligaynon remains Google-first regardless; mBART is last resort.",
        ],
    }

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {args.out}")

    print(
        f"\n=== {report['recommendation']['decision']}: {reason} ===",
        file=sys.stderr,
    )
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
