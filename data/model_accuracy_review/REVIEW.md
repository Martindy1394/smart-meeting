# Model accuracy review — ASR, BART, mBART/NLLB

**Date:** 2026-08-07 (ASR re-check)
**Environment:** CPU; ML deps installed (`requirements-ml.txt`).  
**Machine-readable scores:** `data/model_accuracy_review/report.json`

Reviewed **one by one** against expected outputs (fixtures or labeled clips).

---

## 1. Whisper ASR

**Re-checked:** 2026-08-07 07:09 UTC · 5 trials/clip · median scoring.
See also `ASR_REVIEW.md` for full trial logs.

### Expected vs measured (final pass)

| Clip | Lang | Med WER | Med F1 | Trial pass-rate | Clip pass |
|---|---|---|---|---|---|
| `en_meeting.wav` | en | **0.000** | **1.000** | 100% | ✓ |
| `en_action.wav` | en | **0.000** | **1.000** | 100% | ✓ |
| `tl_greeting.wav` | tl | **0.222** | **0.842** | 80% | ✓ |

**Bar:** median WER ≤ 0.25 or median token-F1 ≥ 0.75
**Verdict: PASS** (mean median WER 0.074)

### Notes

- English final + live: accurate on clean TTS (median WER 0).
- Tagalog: median PASS but **unstable** (80% trial pass; occasional wrong-script tokens).
- Hiligaynon: language routing PASS (auto, never `tl`); proxy TTS median WER 0.455 (weak); native PLD WER still TBD.

---

## 2. BART summarization


### How it works
`llm.summarize(..., source_kind=)` → topic-aware `facebook/bart-large-cnn`:
- `meeting` → frame “Board meeting…” + **Discussion / Decisions / Action items**
- `general` → “Summarize the following.” + flat/topic bullets (no minutes buckets)

### Expected vs measured
Input (English meeting fixture from tests):

> The board approved the quarterly budget after review. Marketing will launch the campaign next week. Facilities will prepare the hall and stage lighting. Security assigned entrance badges for all guests. The chair opened discussion on the annual calendar carefully.

| Check | Expected | Result |
|---|---|---|
| Structure | Discussion, Decisions, Action items | **All three present** |
| Content | budget, Marketing, campaign, calendar | **4/4 (100%)** |
| `general` kind | No Decisions / Action items buckets | **Confirmed** |
| Engine | BART (not extractive-only) | `bart-meeting-minutes` |

**Verdict: PASS**

### Quality caveat
Minutes bullets **duplicate** the frame phrase (“Board meeting discussion and decisions.”) and repeat some action lines. Structure and content coverage pass, but output cleanliness can improve (frame leakage / dedupe inside BART formatting).

---

## 3. Translation (NLLB primary → mBART fallback)

### How it works
`llm.translate` → `_translate_to_english`:
- Tagalog: **NLLB `tgl_Latn`** first (`PH_TRANSLATE_BACKEND=auto`), mBART `tl_XX` fallback  
- Hiligaynon: Google `hil` (unset here) → **NLLB `ceb_Latn`** → mBART last  
- Stock mBART alone: `tl_XX` beats `id_ID` (isolated tag check)

### Expected vs measured (meeting fixtures)

**Tagalog** (`scripts/ph_mt/fixtures/tagalog_en_sample.jsonl`)

| # | token-F1 | Engine | Note |
|---|---|---|---|
| 1–5 short lines | 0.88–1.00 | nllb-200 | Near/exact match to references |
| 6 longer vote line | 0.56 | nllb-200 | Partial / looped |
| 7 action-item line | 0.77 | nllb-200 | Good |

**Mean token-F1 = 0.887** — bar ≥ 0.40 → **PASS**

**Hiligaynon** (`hiligaynon_en_sample.jsonl`, Google unset → NLLB ceb proxy)

| Mean token-F1 | Bar (≥ 0.30) | Weak lines |
|---|---|---|
| **0.724** | PASS | Agenda question 0.31; committee decision 0.27 |

**Stock mBART isolation (Tagalog fixtures, no NLLB)**

| Tag | Mean token-F1 | Historical (2026-07-31) |
|---|---|---|
| `tl_XX` | 0.394 | 0.430 |
| `id_ID` | 0.324 | 0.340 |

Confirms **`tl_XX` > `id_ID`**; production Tagalog path correctly prefers NLLB (much higher F1 than stock mBART alone).

**Verdict:** Tagalog **PASS**; Hiligaynon **PASS on mean** but **NEEDS ATTENTION** on harder lines without Google Translate.

---

## Overall

| Component | Function check | Accuracy vs expected | Verdict |
|---|---|---|---|
| **Whisper ASR** | Final+live load & decode | EN med WER~0.00/0.00; TL med 0.22 (unstable) | **PASS** | EN WER 0; TL WER 0.11 | **PASS** |
| **BART** | Meeting minutes + general kind | Structure + 100% key content | **PASS** (dupe noise) |
| **NLLB/mBART MT** | Three-way route | TL F1 0.89; Hil F1 0.72 | **PASS** (configure Google for Hil) |

### Recommended follow-ups
1. ASR: run WER on real Hiligaynon/Tagalog meeting labels (PLD / board clips).  
2. BART: strip frame-prefix leakage and tighten bullet dedupe.  
3. Hiligaynon MT: enable Google Cloud Translation; keep NLLB/mBART as fallback only.  
4. Do not deploy provisional mBART PH fine-tunes until GPU GO (see prior `data/mbart_ph_ft` work if present on other branches).
