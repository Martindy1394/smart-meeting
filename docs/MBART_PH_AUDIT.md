# mBART Tagalog / Hiligaynon audit (2026-07-31)

Audit of the three-way MT path in `backend/app/services/llm.py`, with
language tags from `languages.py` and line routing from `lang_router.py`.

Re-run the harness:

```bash
cd backend && python3 scripts/audit_mbart_ph_mt.py
python3 scripts/ph_mt/benchmark_mbart_tags.py --out-dir ./data/mt_tag_benchmark
```

## Phase 1 — Execution verification

| Step | Result |
|---|---|
| `lang_router.classify_line` on fixtures | 9/10 correct; 1 thin Hiligaynon line (`Si Maria ang responsable…`) lean-tagged `tl` |
| Real meeting units (ACCO/ADCO) | EN-heavy ASR mix; 1 Hiligaynon-lean clause, 1 Tagalog-lean lyric block |
| `_route_attempts_for_line` (pre-fix) | Tagalog → `[nllb/tl, mbart/id, mbart/en]`; Hiligaynon → `[nllb/hil, mbart/id]` when Google unset |
| `mbart_code` silent mistag | `fil` / `tagalog` / `hiligaynon` / unknown → `None` → historical `or "en_XX"` with **no log** |
| Live `translate()` on fixtures | **10/10 ok** via NLLB-first; **kept_source = 0** |
| Context-window span extraction | Prior fix landed (`_CONTEXT_MIN_UNIT_WORDS`, parity, truncated-span fallback) — not attributed to MT quality below |
| Garbage / Latin heuristics | No false “garbage” flags on fixture hyps; salad-word list still only catches known loops |

**Case (c) kept_source:** `0` on the fixture batch under current NLLB-first defaults.
Google Translate was **not configured** in the audit environment, so Hiligaynon used NLLB `ceb_Latn` rather than Google `hil`.

## Phase 2 — Root causes

### 2a. Tagalog — mislabeled, not unsupported

- mBART-50 vocabulary **includes** `tl_XX` (`tokenizer.lang_code_to_id`).
- Codebase previously forced shorthand `"id"` whenever `MBART_PH_FINE_TUNED_MODEL` was empty.
- Live benchmark on `scripts/ph_mt/fixtures/tagalog_en_sample.jsonl`:

  | Tag | token_f1_mean |
  |---|---|
  | `tl_XX` | **0.430** |
  | `id_ID` | 0.340 |

- Recommendation (applied): stock Tagalog uses native **`tl_XX`**. The `id_ID` path was an inherited workaround, not a measured win.

### 2b. Hiligaynon — genuinely unsupported

- No `hil` / `hil_XX` / Cebuano token in mBART-50 language codes.
- Proxying Hiligaynon as `id_ID` or `tl_XX` is structural degradation, not mistagging.
- Correct order remains: **Google `hil` → NLLB `ceb_Latn` → mBART last resort**.

## Phase 3 — Code + resource plan

### Landed in this change

- `languages.py`: `tl` → `tl_XX`; aliases (`fil`, `tagalog`, `hiligaynon`, …); startup `assert_mbart_codes_resolvable()`.
- `_route_attempts_for_line`: Tagalog mBART always `"tl"`; Hiligaynon mBART documented as degraded last resort.
- `_mbart_translate`: warns on unmapped source codes (no silent English mistag).
- Comments aligned with NLLB’s documented `hil` → `ceb_Latn` proxy.

### Tagalog fine-tune (existing hook)

| Resource | Use |
|---|---|
| FLORES-200 `tgl_Latn` | Held-out chrF/BLEU |
| OPUS (Tatoeba, JW300, GlobalVoices, CCAligned, WikiMatrix) | Parallel pretrain/finetune |
| Domain Taglish / meeting transcripts | Critical for spoken code-switch |
| `MBART_PH_FINE_TUNED_MODEL` + LoRA on **`tl_XX`** | Close residual gap; do not train as `id_ID` |

### Hiligaynon (longer-term)

| Resource | Use |
|---|---|
| SEACrowd, OPUS Bible/JW300 | Sparse parallel text |
| Regional Panay/Negros transcripts | Domain speech |
| New `hil_XX` embedding (init from `tl_XX`/`id_ID`) | Required before fine-tune can “add” the language |
| Google Translate | **Keep primary** until vocabulary work lands |

### Shared eval

- Fixtures: `scripts/ph_mt/fixtures/{tagalog,hiligaynon}_en_sample.jsonl`
- Protocol: `docs/MT_TAG_BENCHMARK.md` + `benchmark_mbart_tags.py`
- Grow a held-out meeting/lyric set with human references; score BLEU/chrF — do not judge quality by spot-check alone.
