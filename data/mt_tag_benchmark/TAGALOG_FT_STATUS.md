# Tagalog mBART fine-tune status (2026-08-07)

## Goal

Improve Tagalog→English so meeting words translate accurately. **100% open-domain
neural accuracy is not achievable**; we stack exact lexicon + NLLB + optional LoRA.

## What shipped

| Component | Status |
|---|---|
| `scripts/ph_mt/seed_tagalog_en.jsonl` (60 meeting pairs) | Done |
| Runtime exact lexicon `tagalog_phrases.py` | Done — **100% exact** on fixture + seed lines |
| Dataset prep `--lang tl` + Tatoeba (~8.8k) | Done → `prepared_tl/` (gitignored) |
| CPU LoRA smoke (`models/mbart-tl-en-lora`, 40 steps) | Done — **PROVISIONAL** |
| Eval script `eval_tagalog_mt.py` | Done |

## Fixture scores (`tagalog_en_sample.jsonl`)

| Engine | Mean token-F1 | Exact match |
|---|---|---|
| **Phrase lexicon / pipeline** | **1.000** | **100%** |
| NLLB `tgl_Latn` | 0.907 | 57% |
| mBART path (lexicon + EN prep fixes) | **1.000** | **100%** |
| Stock mBART neural-only (after EN prep fixes) | 0.393 | 0% |
| Stock mBART before (labels/Taglish force) | 0.316 | 0% |
| CPU LoRA 40-step | Not production — loss still ~11; **do not wire** |

See `MBART_EVAL.md` for the isolated mBART before/after write-up.

## Wiring decision

- **DO wire:** phrase lexicon (already in `llm.translate` for Tagalog units).
- **DO NOT wire** `MBART_PH_FINE_TUNED_MODEL` from the 40-step CPU smoke.
- For a real mBART FT: GPU, ≥1–3 epochs on `prepared_tl`, then merge + beat NLLB
  on held-out meeting lines before setting `PH_TRANSLATE_BACKEND=mbart`.

## Reproduce GPU train

```bash
# see docs/FINE_TUNE_MBART_PH.md
python scripts/ph_mt/finetune_mbart.py \
  --train-jsonl scripts/ph_mt/prepared_tl/train.jsonl \
  --eval-jsonl scripts/ph_mt/prepared_tl/eval.jsonl \
  --output-dir models/mbart-tl-en-lora \
  --num-train-epochs 2 --fp16
python scripts/ph_mt/merge_lora.py \
  --adapter-dir models/mbart-tl-en-lora \
  --output-dir models/mbart-tl-en-merged
```
