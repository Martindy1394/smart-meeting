# mBART language-tag benchmark (tl_XX vs id_ID)

## Why

Stock mBART-50 has **no** Hiligaynon code. Smart Meeting historically used
`id_ID` as a stand-in for both Tagalog and Hiligaynon when no PH fine-tune is
loaded. NLLB is already preferred for PH→EN (`tgl_Latn` / `ceb_Latn`). This
benchmark isolates whether **stock mBART** should prefer native `tl_XX` over
`id_ID` for Tagalog, and collects **human** judgments for Hiligaynon-via-`id_ID`
(the pair with the least safety net).

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

If `tl_XX` mean token-F1 (or chrF) beats `id_ID` by a clear margin on your
domain set, change stock mBART Tagalog mapping from `id_ID` → `tl_XX` in
`languages.py` **for Tagalog only**. Keep Hiligaynon on NLLB-first; only change
Hiligaynon mBART fallback after human eval.

Do **not** invent Hiligaynon quality scores in CI — leave the worksheet for
humans.
