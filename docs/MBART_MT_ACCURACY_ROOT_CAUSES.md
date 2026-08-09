# mBART / NLLB / Google — why transcript translation is inaccurate

**Scope:** Translation stack only (`llm.py`, `languages.py`, `lang_router`,
`google_translate.py`, config).  
**Verified:** 2026-08-07 with runtime probes on this environment.

mBART is only one hop in a three-way router. Stock mBART Tagalog tags are already
correct (`tl_XX`); accuracy still fails because of **wrong Hiligaynon proxies**,
**pre-MT text damage**, **Google unset**, and **ASR garbage in → garbage out**.

---

## Bottom line

| Source lang | What actually runs today | Why accuracy fails |
|---|---|---|
| **Tagalog** | NLLB `tgl_Latn` first (good); mBART `tl_XX` fallback | Pre-MT normalizer **splits Taglish** (`i-move` → broken units); long/mixed lines degrade |
| **Hiligaynon** | Google skipped (not configured) → **NLLB `ceb_Latn`** → mBART `id_ID` | **No native hil code** in NLLB/mBART; Cebuano/Indonesian proxies lose Ilonggo meaning |
| **English** | Passthrough | Spoken normalizer can still shred “because …” clauses before/after MT |

---

## Runtime probe (this host)

| Probe | Result |
|---|---|
| `PH_TRANSLATE_BACKEND` | `auto` → NLLB-first for Tagalog (`prefer_mbart=False`; fine-tune empty) |
| Google | enabled=`true`, **configured=`false`** |
| Hiligaynon route | `[google skipped] → nllb/hil → mbart/id` |
| Tags | `tl`→`tl_XX` / `tgl_Latn`; `hil`→`id_ID` (mBART) / `ceb_Latn` (NLLB) |
| `MBART_PH_FINE_TUNED_MODEL` | empty |
| Taglish normalize | `…nating i-move ang…` → `…nating. i-move. ang…` |
| “because” normalize | `…because of the report and because…` → `… . the report. the committee…` |
| `_nllb_src_code("auto", hil-looking)` | always `tgl_Latn` |

---

## P0 — Blockers

### 1. Hiligaynon has no native MT code; Google unset → Cebuano proxy

**Type:** model limitation + config gap  
**Where:** `llm.py` `_nllb_src_code`, `_ph_translation_attempts`; `languages.py` hil→`id_ID`

```python
# NLLB: hil → ceb_Latn (emergency Visayan proxy)
# mBART: hil → id_ID (degraded typological proxy)
# Primary intended path: Google Cloud Translation language="hil"
```

Google is enabled in config but **credentials are not configured**
(`google_translate.is_configured() == False`), so every Hiligaynon unit falls
through to NLLB Cebuano then Indonesian mBART.

**Why it hurts:** Ilonggo is not Cebuano or Indonesian. Prior review: Hil mean
token-F1 ~0.72 overall, with agenda/committee lines ~0.27–0.31.

---

### 2. `_normalize_spoken_transcript` destroys Taglish before MT

**Type:** code bug  
**Where:** `llm.py` `_ENGLISH_AFTER_PH_RE` (case-insensitive `I|…`) +
`_normalize_spoken_transcript` applied at translate entry (~1825)

Tagalog/Taglish **`i-` verb prefixes** match English “I”:

```
SRC : Pwede ba nating i-move ang botohan sa susunod na linggo ...
NORM: Pwede ba nating. i-move. ang botohan sa susunod na linggo ...
```

The model then translates **three broken fragments** instead of one sentence
(matches weak long Tagalog fixture F1 ~0.56).

Same normalizer also **drops English discourse**:

```
SRC : We need this because of the report and because the committee is late.
NORM: We need this. the report. the committee is late.
```

Re-applied on English after MT inside summarize → minutes lose causal links.

---

## P1 — High

### 3. ASR errors enter MT with almost no repair

Finalize stores Whisper text (hallucination collapse only). Translate collapses
loops again, then MT. No PH lexical repair.

Hiligaynon ASR proxy median WER ~0.45 (see `ASR_REVIEW.md`). Wrong words in →
wrong English out. MT cannot invent the correct Ilonggo form.

### 4. Stock mBART ≪ NLLB for Tagalog; fine-tune slot empty

Under `auto` + empty `MBART_PH_FINE_TUNED_MODEL`, Tagalog correctly prefers NLLB
(~0.89 F1 vs stock mBART `tl_XX` ~0.39). If `PH_TRANSLATE_BACKEND=mbart` without
a strong checkpoint, Tagalog accuracy collapses. Provisional PH LoRA must not be
wired until eval GO (`docs/FINE_TUNE_MBART_PH.md`).

### 5. Unknown/auto NLLB source always forced to `tgl_Latn`

```python
# _nllb_src_code: both branches return "tgl_Latn"
```

Hiligaynon-looking auto text still tagged Tagalog inside NLLB → wrong lexicon.

### 6. `llm._FILIPINO_MARKERS` is Tagalog-only

Markers omit `gid/indi/sang/kag/subong/…`. Hiligaynon line scores `(en=0, fi=0)`
in llm helpers → weak garbage/coverage decisions (lang_router has a better hil list).

---

## P2 — Medium amplifiers

| # | Issue | Effect |
|---|---|---|
| 7 | Context window dies after first unit | Later units get EN context only; cross-lang guard skips; Google is line-local |
| 8 | Silent truncation (`truncation=True`, max 512/1024) | Long unpunctuated ASR run-ons lose tails |
| 9 | Non-EN `translate()` auto still picks `id` when `fi≥0.08` | Filipino mistagged Indonesian off the EN path |
| 10 | Fine-tune proxies hil under `tl_XX` | Cannot add `hil_XX`; Google must stay primary for Hiligaynon |
| 11 | Glossary placeholders | Usually OK; many names can confuse NLLB/mBART |

---

## What is *not* the main Tagalog→EN problem today

- Using `id_ID` for Tagalog on the English path — **fixed** (`tl_XX`).
- Preferring stock mBART over NLLB under current `.env` — **auto correctly prefers NLLB**.

mBART is the **fallback / last resort**, not the primary Tagalog engine. Blaming
“mBART alone” misses NLLB + Google routing and the pre-MT normalizer.

---

## Transcript → translation damage chain

```
Whisper final_transcript
  → (optional) glossary protect
  → _normalize_spoken_transcript   ← Taglish / "because" damage
  → idea-unit split
  → lang_router (en / tl / hil)
  → EN passthrough
    | TL → NLLB tgl_Latn → mBART tl_XX
    | HIL → Google hil (missing) → NLLB ceb_Latn → mBART id_ID
  → reassemble + faithfulness warnings
  → summarize may re-normalize English
```

---

## Fix status (implemented 2026-08-07)

1. **Ops still required:** Configure Google Translate credentials for Hiligaynon.
2. **Done:** Taglish `i-` prefixes not split as English “I”; hyphenated stems kept
   intact before `ang`/`mga`; “because” markers kept (not deleted); post-MT
   English uses `_normalize_mt_english` (no spoken PH splitter).
3. **Done:** Auto NLLB source uses `lang_router` (hil → `ceb_Latn`, tl → `tgl_Latn`);
   Hiligaynon markers added to `_FILIPINO_MARKERS`; non-EN auto mBART prefers `tl`
   over `id`.
4. **Still needed:** Hiligaynon ASR fine-tune + Google for true hil accuracy.
5. PH mBART LoRA remains Tagalog-oriented only (cannot add `hil_XX`).

---

## Related artifacts

- Accuracy numbers: `data/model_accuracy_review/REVIEW.md` / `report.json`
- Tag audit: `docs/MBART_PH_AUDIT.md`
- Paragraph/context review: `docs/MBART_PARAGRAPH_CONTEXT_REVIEW.md`
- Fine-tune: `docs/FINE_TUNE_MBART_PH.md`
- ASR feed quality: `docs/ASR_PH_ACCURACY_ROOT_CAUSES.md`
