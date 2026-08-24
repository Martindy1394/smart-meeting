# mBART Hiligaynon evaluation (isolated)

Scope: **mBART path only** (`_mbart_translate` / Hiligaynon lexicon / cognate
bridge / dual-tag fallback). NLLB, Google Translate, Whisper, and BART were not
changed for this pass.

## Problem

Whisper often captures Hiligaynon as Tagalog-leaning noise (`naton`→`natin`,
`indi` kept beside Tagalog particles, etc.). That text is then encoded with
mBART **`tl_XX`**. Stock mBART has **no `hil_XX`**; the hil proxy is degraded
**`id_ID`**.

## Fixture sets

| Set | File |
|---|---|
| Clean Hiligaynon | `scripts/ph_mt/fixtures/hiligaynon_en_sample.jsonl` |
| Hiligaynon as Tagalog ASR noise | `scripts/ph_mt/fixtures/hiligaynon_as_tagalog_noise.jsonl` |
| Tagalog control | `scripts/ph_mt/fixtures/tagalog_en_sample.jsonl` |

## Neural-only results (lexicons disabled)

| Condition | Before | After (cognate + dual-tag) |
|---|---|---|
| hil_clean → `id_ID` | 0.235 | ~0.24 (still weak stock proxy) |
| hil_clean → `tl_XX` (mistagged) | 0.284 | **0.333** |
| hil_as_tl_noise → `tl_XX` | 0.249 | **0.407** |
| hil_as_tl_noise → `id_ID` | 0.244 | **0.287** |
| tagalog → `tl_XX` control | 0.393 | 0.393 (unchanged) |

Artifacts: `mbart_hiligaynon_eval_neural.json`,
`mbart_hiligaynon_eval_after_neural.json`.

## Full mBART path (lexicons on)

Curated Hiligaynon + Tagalogized-noise + Tagalog meeting lines short-circuit
via exact lexicons → **mean token-F1 1.000 / exact 100%** on these fixtures
(`mbart_hiligaynon_eval_after.json`).

## mBART-only improvements shipped

1. `hiligaynon_phrases.py` — exact Hiligaynon / noise-fixture lexicon before decode
2. `mbart_hiligaynon.py` — Hiligaynon cue detect + cognate bridge for `tl_XX`
3. Dual-tag fallback inside `_mbart_translate` (`tl_XX` ↔ `id_ID`) when a
   Hiligaynon-heavy line is rejected as garbage
4. Eval harness: `scripts/ph_mt/eval_mbart_hiligaynon.py`

## Reproduce

```bash
# Neural-only ablation
python scripts/ph_mt/eval_mbart_hiligaynon.py --neural-only \
  --out data/mt_tag_benchmark/mbart_hiligaynon_eval_after_neural.json

# Full mBART path (lexicons)
python scripts/ph_mt/eval_mbart_hiligaynon.py \
  --out data/mt_tag_benchmark/mbart_hiligaynon_eval_after.json
```

## Limits

- Stock neural mBART remains weak on open Hiligaynon; Google → NLLB stay the
  preferred Hiligaynon route in `PH_TRANSLATE_BACKEND=auto`.
- These fixes help when the **mBART engine** runs (fallback / `backend=mbart`)
  on mistagged or Tagalogized Hiligaynon.
- No `hil_XX` vocabulary slot can be invented without a new model.
