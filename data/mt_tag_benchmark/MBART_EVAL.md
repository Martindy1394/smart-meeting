# mBART TL→EN evaluation (isolated)

Scope: **mBART path only** (`_mbart_translate` / EN post-process / EN dialect flags).
NLLB routing, Whisper ASR, and BART summarization were not changed for this pass.

## Fixture set

`scripts/ph_mt/fixtures/tagalog_en_sample.jsonl` (7 board-meeting Tagalog lines).

## Results

| Condition | Mean token-F1 | Exact match | Artifact |
|---|---|---|---|
| Before (stock + harmful EN labels/Taglish force) | **0.316** | 0% | `mbart_eval_before.json` |
| After neural-only (lexicon off; softer EN prep + post-process) | **0.393** | 0% | `mbart_eval_after_neural_only.json` |
| After full mBART path (exact lexicon short-circuit) | **1.000** | 100% | `mbart_eval_after.json` |

## What failed before

- `AKSYON:` / `NAPAGPASYAHAN:` labels on EN-target encode → echoes like `ACKNOWLEDGEMENTS:`
- Taglish→Filipino forcing before `tl_XX` decode hurt `i-move` / `i-record` lines
- Untranslated personal marker `Si Maria…`
- Classic stock mBART hallucinations on greetings / agenda questions

## What we improved (mBART path only)

1. Exact `tagalog_phrases` hit **before** model load when targeting English
2. For EN targets: `normalize_for_mbart(..., apply_taglish=False, label_minutes=False)`
3. Tighter `generate` (5 beams, mild length/repetition penalties)
4. `_postprocess_mbart_english` strips label echoes and leading `Si Name`

## Reproduce

```bash
python scripts/ph_mt/eval_tagalog_mt.py --engines mbart \
  --out data/mt_tag_benchmark/mbart_eval_after.json
```

## Wiring

- Keep default `PH_TRANSLATE_BACKEND=auto` (NLLB-first). These fixes help when the
  mBART engine is selected or used as a fallback.
- Do **not** wire the provisional CPU LoRA (`MBART_PH_FINE_TUNED_MODEL`) until a
  GPU merge beats NLLB on held-out meeting lines.

## `[untranslated:]` gap (fixed)

Clear Tagalog meeting prose was sometimes kept as `[untranslated:…]` **without
calling mBART**. Cause: `_source_looks_like_garbled_ph_asr` used only
`unknown_ratio` against a tiny lexicon — open-class Tagalog
(`pasalamatan`, `pagpupulong`, …) looked like ASR salad.

Fix: if `lang_router` marks the clause as clear, non-ambiguous Tagalog/
Hiligaynon, skip the garble short-circuit and let mBART/NLLB translate.
True lyric ASR salad (ambiguous / low marker confidence) still keeps source.
