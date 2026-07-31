# mBART language-tag benchmark (tl_XX vs id_ID)

## Why

Stock mBART-50 has **no** Hiligaynon code. Smart Meeting historically used
`id_ID` as a stand-in for both Tagalog and Hiligaynon when no PH fine-tune is
loaded. A 2026-07-31 live run preferred **`tl_XX` over `id_ID` for Tagalog**
(token-F1 0.43 vs 0.34) — stock Tagalog now maps to `tl_XX` (see
`docs/MBART_PH_AUDIT.md`). NLLB remains preferred for PH→EN by default
(`tgl_Latn` / `ceb_Latn`). This benchmark still isolates stock mBART tag choice
and collects **human** judgments for Hiligaynon-via-`id_ID` (degraded last
resort only).

## Run

```bash
# Protocol + worksheets (no GPU / no model download)
python3 scripts/ph_mt/benchmark_mbart_tags.py --fixtures-only \
  --out-dir ./data/mt_tag_benchmark

# Live comparison (transformers + torch + model weights)
python3 scripts/ph_mt/benchmark_mbart_tags.py \
  --out-dir ./data/mt_tag_benchmark
```

Optional: `pip install sacrebleu` for chrF++ sentence scores.

## Outputs

| File | Purpose |
|---|---|
| `report.json` | Aggregate token-F1 / chrF + Tagalog recommendation |
| `tagalog_tl_XX.jsonl` / `tagalog_id_ID.jsonl` | Per-sentence hyps |
| `hiligaynon_id_ID.jsonl` | Model English for Hiligaynon via `id_ID` |
| `hiligaynon_id_ID_human_eval.md` | Adequacy/fluency sheet for bilingual raters |

## Decision rule (Tagalog)

**Applied:** stock Tagalog → `tl_XX` in `languages.py` after the fixture
benchmark win. Re-run on a larger domain (meeting Taglish) set before changing
again. Keep Hiligaynon on Google → NLLB; only change Hiligaynon mBART fallback
after human eval — it remains a degraded option with no vocabulary slot.

Do **not** invent Hiligaynon quality scores in CI — leave the worksheet for
humans.
