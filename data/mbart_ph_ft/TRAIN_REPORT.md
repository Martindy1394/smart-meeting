# mBART PH fine-tune — smoke-run report (2026-07-31)

Environment: CPU-only cloud agent (no CUDA). Checkpoints under `models/`
(gitignored). Eval JSON committed beside this file.

## Checkpoints produced

| Lang | Adapter | Merged | Token |
|---|---|---|---|
| Tagalog | `models/mbart-tl-en-lora` | `models/mbart-tl-en-merged` | native `tl_XX` |
| Hiligaynon | `models/mbart-hil-en-lora` | `models/mbart-hil-en-merged` | **new** `hil_XX` (init from `tl_XX`) |

Train recipe (smoke):

- Tagalog: LoRA r=16/α=32 on `q_proj`/`v_proj`, `--max-steps 60`, Tatoeba (~8.3k) + domain seeds
- Hiligaynon: vocab extend + unfrozen shared embed (`--embed-lr 5e-4`) + LoRA, `--max-steps 40`, 57 seed pairs

FLORES-200 was **not** available (gated Hub dataset without `HF_TOKEN`).

## Domain meeting-fixture scores

### Tagalog (`fixtures/tagalog_en_sample.jsonl`, n=7)

| System | token-F1 | BLEU | chrF |
|---|---|---|---|
| Fine-tune (`tl_XX`) | **0.480** | 16.3 | 44.0 |
| Stock `tl_XX` | 0.384 | 11.8 | 39.9 |
| Stock `id_ID` | 0.318 | 12.0 | 33.4 |

Decision: **PROVISIONAL** — clear metric lift over both baselines, but below
production floors (F1≥0.55 / BLEU≥20). **Do not wire** into
`MBART_PH_FINE_TUNED_MODEL` yet.

### Hiligaynon (`fixtures/hiligaynon_en_sample.jsonl`, n=7)

| System | token-F1 | BLEU | chrF |
|---|---|---|---|
| Fine-tune (`hil_XX`) | **0.354** | 18.1 | 41.0 |
| Stock `tl_XX` mistag | 0.297 | 4.1 | 33.0 |
| Stock `id_ID` mistag | 0.235 | 4.7 | 26.9 |

Decision: **PROVISIONAL** — `hil_XX` vocab path works and beats mistag
baselines, but F1 floor (0.40) not met; data too scarce for a smoke run.
**Keep Google Translate primary.** Do not wire yet.

## Go / no-go summary

| Checkpoint | Metric improvement | Production GO | Wire `MBART_PH_FINE_TUNED_MODEL`? |
|---|---|---|---|
| Tagalog merged | Yes vs `tl_XX`/`id_ID` | **NO** | **No** |
| Hiligaynon merged | Yes vs mistags + real `hil_XX` | **NO** | **No** |

### Next steps for a real GO

1. GPU train Tagalog ≥1–3 epochs on full Tatoeba + OPUS-100 + authenticated FLORES holdout + more meeting/lyric pairs.
2. Collect more Hiligaynon bitext (SEACrowd / OPUS Bible/JW300 / regional transcripts) via `--extra-jsonl`; retrain with `--lang hil` for several epochs.
3. Re-run `evaluate_mbart_checkpoint.py` until decision=`GO`.
4. Only then set `MBART_PH_FINE_TUNED_MODEL` (Tagalog and/or Hiligaynon merged path).

Full machine-readable scores: `tl_eval_report.json`, `hil_eval_report.json`.
Training how-to: `docs/FINE_TUNE_MBART_PH.md`.
