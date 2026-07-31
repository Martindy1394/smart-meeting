# mBART paragraph-context structural review

**Scope:** mBART-only (`_mbart_translate`, `_Pipelines.mbart`, `mbart_code`,
`MBART_MODEL` / `MBART_PH_FINE_TUNED_MODEL`, and mBART-facing cleanup helpers).
Frozen code was read for context only and is **not** edited here.

**Active checkpoint (this deployment):**
`facebook/mbart-large-50-many-to-many-mmt`

`settings.mbart_ph_finetuned_model` is empty, so `_Pipelines.mbart()` loads
stock mBART-50 many-to-many — **not** a PH fine-tune.

---

## Phase 1 — Structural map

```text
source text
    │
    ├─ translate(tgt!="en") ──► _mbart_translate(text, src, tgt)
    │                              src from caller / auto:
    │                              Filipino auto → "id" (see tag table)
    │
    └─ _translate_to_english ──► per idea-unit
            │
            └─ _route_attempts_for_line  (frozen; read-only)
                    passes shorthand src ∈ {tl, id, en, …}
                    │
                    └─ _translate_unit_with_context  (frozen; read-only)
                           window = prev_units[-2:] + unit   ← ONLY multi-unit input
                           │
                           └─ _mbart_translate(window|unit, src, "en")
                                  │
                                  ├─ mbart_code(src) → tl_XX / id_ID / en_XX / …
                                  ├─ mbart_code(tgt) → en_XX (EN path)
                                  ├─ _chunk_text(size=3500, overlap=280 if long)
                                  │     each chunk → independent model.generate()
                                  │     tokenize: truncation=True, max_length=1024
                                  │     beams=6, no_repeat_ngram_size=4,
                                  │     length_penalty=1.05,
                                  │     max_new_tokens=min(512, max(64, words*2.8))
                                  ├─ decode → _strip_leaked_lang_codes
                                  │         → _collapse_translation_loops
                                  └─ if tgt == en_XX only:
                                        _looks_like_latin_script
                                        _is_garbage_english_translation
                                        (raise _NonEnglishTranslation → upstream retry)
                                  join chunk outputs with " "
```

### Model loading

| Setting | Value here |
|---|---|
| `mbart_model` | `facebook/mbart-large-50-many-to-many-mmt` |
| `mbart_ph_finetuned_model` | `""` (inactive) |
| **Loaded id** | stock mBART-50 |

### Language-tag resolution (`mbart_code`)

Every shorthand actually passed into mBART from callers, checked against
`tokenizer.lang_code_to_id`:

| Shorthand (callers) | Resolves to | In vocab? |
|---|---|---|
| `en` | `en_XX` | yes |
| `tl`, `fil`, `tagalog` | `tl_XX` | yes |
| `id` | `id_ID` | yes |
| `hil`, `hiligaynon`, `ilonggo` | `id_ID` (degraded) | proxy only — **no `hil_XX`** |
| `es`…`ko` (LANGUAGES targets) | matching `*_XX` / `*_DE` / … | yes |

**Call sites into `mbart_code`:**

1. `_mbart_translate(src_code)` / `_mbart_translate(tgt_code)`
2. Indirect: `_translate_unit_with_context` → `_mbart_translate(..., src, "en")`
   where `src` comes from `_route_attempts_for_line`:
   - Tagalog: always `"tl"` → `tl_XX`
   - Hiligaynon last resort: `"id"` (or `"tl"` if PH fine-tune set) — never a real Hiligaynon token
   - Unknown/mixed: `"tl"`, then `"en"`
3. `translate()` non-English target: auto Filipino score ≥ 0.08 sets `src = "id"`
   → **`id_ID`**, not `tl_XX` (Tagalog→non-EN path still uses the old Indonesian tag)

Unmapped codes: `_mbart_translate` logs a warning and emergency-falls back to
`id_ID` (no silent `en_XX` mistag).

### Generation path (`_mbart_translate`)

1. Resolve src/tgt tags; identity `en_XX→en_XX` returns text unchanged.
2. Lock (`mbart_infer_lock`) for tokenize+generate.
3. `_chunk_text`: sentence-aware, `size=3500`; if `len(text) > 3500`,
   `overlap_chars=280` (overlap is **baked into subsequent source chunks**).
4. Per chunk: `truncation=True, max_length=1024` (silent), then `generate` with
   the params above.
5. Cleanup always: leak strip + loop collapse.
6. EN-only gates: Latin script + garbage detector; failure raises
   `_NonEnglishTranslation` (aborts that call; upstream may retry another engine).
7. Chunk strings joined with a single space — **no target-side overlap trim,
   no decoder state across chunks**.

### How mBART ever sees more than one idea-unit

**Only** via `_translate_unit_with_context`:
`window = " ".join(prev_units[-context_n:] + [unit])` then
`_mbart_translate(window, src, "en")`.

There is no paragraph-/document-level cache, beam state, or memory inside
`_mbart_translate` itself. Long single strings are the other way mBART sees
multi-sentence text (direct `translate` / window), and those are still subject
to chunking + independent generates.

---

## Phase 2 — Hypotheses (confirmed / ruled out)

### 2a. Per-chunk generation has no shared state — **CONFIRMED**

**Structural proof (code):** the generate loop in `_mbart_translate` calls
`model.generate(**encoded, …)` once per chunk with a fresh encoding. Nothing
passes past decoder hidden state, forced tokens, or prior English output into
the next iteration. Continuity is source-overlap text only (`overlap_chars=280`
inside `_chunk_text`). Outputs are `" ".join(outputs)`.

**Measurement (2026-07-31, stock mBART-50, CPU):**

- Multi-chunk Tagalog meeting-style input (`chars=3915` → 2 chunks;
  source overlap ≈ 273 chars).
- Chunk 0: no `Maria`; English out had `Maria=0`, `she=0`, `he=0`.
- Chunk 1: contains `Maria` + `siya`; English out had `Maria=2`, still
  `she=0` / `he=0` (stock quality weak, but the **entity never crosses into
  chunk 0’s generation** because that chunk never saw it).
- Pronoun unit alone vs with antecedent:
  - Alone (`Sinabi niya…`): no Maria / no she.
  - Window (`Si Maria Cruz…` + unit): Maria present in the English window.
  - That gap is exactly “no discourse state unless the source string itself
    still contains the antecedent.”

**Side effect:** if one chunk trips the EN garbage/Latin gate, the **entire**
`_mbart_translate` call raises — there is no partial commit of good chunks.

**Overlap join:** source overlap is re-translated in chunk N+1 and concatenated.
On this run, near-join 4-gram duplication was not observed (model drift across
chunks differed), but the mechanism for duplicated or inconsistent renderings
of the same source span remains in place.

### 2b. Silent truncation at 1024 tokens — **CONFIRMED (reachable edge); ruled out for normal prose chunks**

| Input shape | Chunks | Tokens / chunk | Truncates? |
|---|---|---|---|
| Sentence-aware ~3500-char Tagalog | 2 | 770 / 575 | **no** |
| Meeting pad multi-chunk | 2 | 851 / 174 | **no** |
| Undividable `"salita "*1200` (no `.?!`) | 1 | **1202** | **yes** |
| Single huge sentence (~7200 chars) | 1 | **1207** | **yes** |

So: with current `_MAX_CHUNK_CHARS=3500` and sentence breaks, practice stays
under 1024. Without sentence boundaries (ASR run-ons, lyric blocks, missing
punctuation), `_chunk_text` returns one oversized chunk and
`truncation=True, max_length=1024` drops the tail **with no log**. That is an
independent, undetected paragraph-context loss path.

### 2c. Paragraph context is bolted on, not structural — **CONFIRMED** (with frozen-helper confound)

- `_mbart_translate` has no notion of “current unit vs context.”
- The only multi-unit path is shared `_translate_unit_with_context` (also used
  by NLLB/Google dispatch) — not sized to mBART’s 1024-token budget or
  chunk/overlap behavior.

**Samples (stock mBART; quality aside, behavior of the wiring):**

| Case | Alone | Window (mBART on full window) | Pipeline span (`_extract_target_span`) |
|---|---|---|---|
| Pronoun + Ana antecedent | No Ana | Ana appears | Span kept Ana (also kept context sentence) |
| Short unit `"Oo."` | Alone path | — | Context **skipped** (`_CONTEXT_MIN_UNIT_WORDS=6`) |
| Pedro/Juan + `niya/siya` | No names | Names in window | **Names dropped from span** |

So: giving mBART a window *can* change the hypothesis vs unit-alone, but
whether the pipeline **keeps** that improvement depends on
`_extract_target_span` / parity / truncation guards — tracked as a **separate
task**. Do not conclude “context doesn’t help mBART” from span-path failures
alone; measure `window_full` (as above) when judging the model path.

### 2d. Tagalog `tl_XX` vs `id_ID` on longer context — **CONFIRMED: `tl_XX` still wins; gap does not collapse**

Fixture set: `scripts/ph_mt/fixtures/tagalog_en_sample.jsonl` (stock mBART-50,
same generate hyperparams as `_mbart_translate`).

| Bucket | n | tl_XX F1 | id_ID F1 | Δ (tl−id) |
|---|---|---|---|---|
| short (≤12 words) | 5 | 0.379 | 0.316 | **+0.062** |
| medium (13–30) | 2 | 0.398 | 0.322 | **+0.076** |
| all 7 lines joined (65 words) | 1 | 0.493 | 0.328 | **+0.165** |

Direction matches the earlier audit (0.43 vs 0.34). On this sample the gap
**widens** slightly from short → medium → joined paragraph — it does **not**
narrow with multi-sentence context. (Small n; treat as supporting evidence,
not a new official benchmark.)

Synthetic length sweep (meeting-style paragraph ×1 / ×3) also showed Δ moving
from near-zero/negative on a single awkward short line toward **positive** for
`tl_XX` as length grew.

### 2e. Hiligaynon has no vocabulary slot — **CONFIRMED (not a context bug)**

- `"hil_XX" in tokenizer.lang_code_to_id` → **False** (no `hil*` keys at all).
- `mbart_code("hil")` → `id_ID` (degraded); with PH fine-tune it would return
  `tl_XX` — still not Hiligaynon.
- Proxy translations of a Hiligaynon paragraph under `id_ID` / `tl_XX` are
  English-looking but not reliable Hiligaynon→EN.

Failures here are the **vocabulary ceiling** from the prior PH audit, not
paragraph-context loss. Out of scope for context fixes; keep Google → NLLB →
mBART-last routing.

---

## Phase 3 — Fix plan (proposal only — not implemented)

Ranked by expected impact vs effort. All edits must stay inside the in-scope
list unless marked **DEPENDENCY**.

### P0 — Detect / avoid silent 1024 truncation (2b)

| | |
|---|---|
| **Where** | `_mbart_translate` (and optionally tighten `_chunk_text` sizing used only by mBART) |
| **Change** | Before generate: if `encoded["input_ids"].shape[-1] >= 1024` (or tokenizer reports truncation), log a warning with chunk preview; prefer splitting on commas/semicolons/whitespace when sentence split yields one oversized chunk so tokens stay &lt; ~960. |
| **Effect** | Removes undetected tail loss on ASR run-ons / unpunctuated blocks. |
| **Verify** | Input = `"salita " * 1200` (today: 1202 tokens, silent truncate). After: ≥2 chunks, each &lt;1024 tokens, warning logged if still forced to truncate; English output covers content from both halves (spot-check last words of source appear in some chunk’s translation). |

### P1 — Target-side continuity / overlap join (2a)

| | |
|---|---|
| **Where** | `_mbart_translate` only |
| **Change** | After translating chunk N+1, drop the English prefix that likely re-translates the source overlap (e.g. align by source-overlap word ratio, or skip re-decoding the overlapped source span). Optionally prepend the last 1–2 **English** sentences from chunk N as a decoder hint only if a clean API exists without changing frozen callers — otherwise keep to source-overlap dedupe on join. |
| **Effect** | Less duplicated / inconsistent renderings across chunk boundaries; fewer aborted full-calls when only the overlapped salad region is bad. |
| **Verify** | Same 3915-char multi-chunk fixture: count shared 4-grams near the join before/after; entity block at end of text must still appear once in the joined English; one-chunk garbage must not discard a clean second chunk if you add partial-commit (optional stretch). |

### P2 — Tagalog auto-tag on non-EN `translate()` (language fidelity)

| | |
|---|---|
| **Where** | `translate()` branch that sets `src = "id" if fi >= 0.08 else "en"` before `_mbart_translate` — **in-scope** as an mBART call-site tag bug |
| **Change** | Use `"tl"` (→ `tl_XX`) for Filipino-detected source when calling mBART, matching the EN-path router. |
| **Effect** | Non-English targets stop mistagging Tagalog as Indonesian. |
| **Verify** | `translate("Kailangan nating aprubahan ang budget…", target_language="es", source_language="auto")` — log/trace `tokenizer.src_lang == "tl_XX"`; compare token-F1 / manual quality vs forced `id`. |

### P3 — mBART-aware context window — **DEPENDENCY (frozen)**

| | |
|---|---|
| **Gap** | 2c: window helps sometimes, but span extract / short-unit skip / parity live in `_translate_unit_with_context` / `_extract_target_span` (frozen; separate task). |
| **In-scope alone** | Could add a thin `_mbart_translate_with_budget(window)` helper that truncates **context prefix** (not the current unit) to fit &lt;1024 tokens — but callers that build `window` are frozen, so wiring it needs the other task. |
| **Do not fix here** | `_extract_target_span`, sentence-parity, short-unit threshold. |
| **Verify (after dependency lands)** | Pronoun fixture: alone lacks Ana; `window_full` has Ana; pipeline span keeps the unit’s English **and** the gender/name cue without swallowing the whole window or dropping the unit. |

### P4 — Tagalog `tl_XX` on long inputs (2d)

| | |
|---|---|
| **Status** | Mostly **already landed** (`mbart_code("tl")` → `tl_XX`; router passes `"tl"`). |
| **Residual** | Extend `benchmark_mbart_tags.py` (or a small script) with length-stratified + joined-paragraph rows so regressions show Δ by length. No production code change required unless P2’s non-EN path is fixed. |
| **Verify** | Re-run length table in Phase 2; expect Δ(tl−id) ≥ 0 on medium/joined buckets. |

### P5 — Hiligaynon (2e)

| | |
|---|---|
| **Verdict** | Not a paragraph-context fix. No in-scope code change. |
| **Pointer** | `docs/MBART_PH_AUDIT.md` — Google → NLLB → mBART last resort; new `hil_XX` embedding is a model/vocab project. |

### Priority summary

| Rank | Fix | Impact | Effort | Scope |
|---|---|---|---|---|
| P0 | Truncation detect + hard-split oversized chunks | High (silent loss) | Low–med | in-scope |
| P1 | Overlap join / per-chunk continuity | High on long inputs | Med | in-scope |
| P2 | `translate()` auto src `id`→`tl` | Med (non-EN path) | Low | in-scope |
| P3 | mBART-aware context + span | High for unit MT | Med | **DEPENDENCY** on frozen helpers |
| P4 | Length-stratified `tl_XX` eval | Low (monitoring) | Low | scripts/docs |
| P5 | Hiligaynon vocab | N/A here | — | prior audit |

---

## Explicit non-edits (frozen)

Not modified in this review task:

- `_pipelines.nllb`, `_nllb_translate_to_english`, `_nllb_src_code`
- `google_translate.py`
- BART summarization (`_bart_summarize_chunk`, …)
- `lang_router.py`, `_route_attempts_for_line` (engine order)
- `_extract_target_span`, `_translate_unit_with_context` (referenced only; P3 dependency)

---

## How to reproduce measurements

```bash
cd /workspace/backend
# Active checkpoint
python3 -c "from app.config import settings; print((settings.mbart_ph_finetuned_model or '').strip() or settings.mbart_model)"

# Token budget / truncation reachability, multi-chunk generate, tl vs id by length:
# (ad-hoc scripts used 2026-07-31; see Phase 2 tables above)
# Fixture benchmark:
python3 ../scripts/ph_mt/benchmark_mbart_tags.py --out-dir ../data/mt_tag_benchmark
```
