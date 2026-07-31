# Smart Meeting — model summaries

Revised edition — audience split, decision rationale, merged performance gap  
Reviewed: July 24, 2026

## Contents

1. [Who this is for](#who-this-is-for)
2. [Executive summary](#executive-summary)
3. [Why these models (and what we rejected)](#why-these-models-and-what-we-rejected)
4. [Performance, latency, and cost](#performance-latency-and-cost)
5. [Whisper — speech recognition](#whisper--speech-recognition)
6. [BART — English meeting minutes](#bart--english-meeting-minutes)
7. [mBART — translation (with NLLB for PH→EN)](#mbart--translation-with-nllb-for-phen)
8. [How they fit together](#how-they-fit-together)
9. [Glossary](#glossary)
10. [Review follow-ups (de-duplicated)](#review-follow-ups-de-duplicated)

---

## Who this is for

One structural issue (not two): the older summary assumed readers already
knew Whisper / BART / mBART, yet stayed too thin for implementers.

| Reader | Read this | Then |
|---|---|---|
| **Product / PM** | [Executive summary](#executive-summary), [How they fit together](#how-they-fit-together), [Glossary](#glossary) | Decide product questions (Hiligaynon bias, English minutes, optional languages) without env-var detail |
| **Engineer / operator** | [Why these models](#why-these-models-and-what-we-rejected), each model’s **Settings / Key files / Docs**, [Performance…](#performance-latency-and-cost) | Wire `.env`, fine-tune, or change backends against the linked scripts |

This file is the **map**. Implementation depth lives in the linked docs and
code paths below — not duplicated here as a second incomplete guide.

---

## Executive summary

Smart Meeting turns board-meeting audio into minutes with three roles:

| Owner | Model family | Job |
|---|---|---|
| Accuracy of what was said | **Whisper** (+ optional Tagalog RNN-T live) | Audio → transcript |
| Language access | **NLLB** (PH→EN default) + **mBART** (many-to-many / fallback) | Transcript → English (and other languages) |
| Readable minutes | **BART** | Flat pass on English → Discussion / Decisions / Action items (no topic re-segmentation of the translation) |

Pipeline:

```
Live PCM  → Whisper (live captions)
Full WAV  → Whisper (final transcript)
Transcript → NLLB / mBART → English → BART → meeting minutes
Transcript → mBART → other languages (on demand)
```

Meetings are **Hiligaynon-biased**. Whisper has no native `hil` token, so
decode stays on **auto-detect** (never forced Tagalog `tl`). Minutes are
always produced from **English**, not raw PH speech.

---

## Why these models (and what we rejected)

Rationale below is from **in-repo defaults and docs**, not fabricated
benchmarks. Where we lack measured numbers, that is called out in
[Performance, latency, and cost](#performance-latency-and-cost).

### Whisper for ASR

| Choice | Why (project fact) | Alternatives considered / deferred |
|---|---|---|
| Two-pass Whisper (live `small`, final `medium` / HF PH) | Live needs low latency; final needs accuracy for minutes | Single-pass only — rejected (captions either lag or stay rough) |
| `rbcurzon/whisper-medium-ph` as Hiligaynon/Visayan HF fallback | Stock Whisper is weak on Ilonggo; PH-medium is the best public dialect checkpoint we wire by default | Forcing Tagalog `tl` for Hiligaynon — **rejected** (mismatch caused bad transcripts) |
| Optional Tagalog RNN-T live only | Lower-latency streaming when NeMo is installed; no public Hiligaynon RNNT | Replacing Whisper final with RNN-T — deferred (final stays Whisper) |
| PLD fine-tune path | ~41h labeled Hiligaynon in UP-DSP PLD | Waiting on a public Hiligaynon Whisper checkpoint that does not exist yet |

### NLLB + mBART for translation

| Choice | Why (project fact) | Alternatives considered / deferred |
|---|---|---|
| NLLB-200 distilled first for PH→EN (`PH_TRANSLATE_BACKEND=auto`) | Better Tagalog codes (`tgl_Latn`); Hiligaynon → Google `hil` then `ceb_Latn` | Stock mBART alone — Tagalog uses native `tl_XX`; Hiligaynon has no token (degraded `id_ID` last) |
| mBART-50 many-to-many for non-EN targets | Already supports the UI language list | Running NLLB for every target — not wired; mBART owns non-EN |
| Optional `MBART_PH_FINE_TUNED_MODEL` | Teams that want a local PH→EN mBART LoRA | NLLB fine-tune — documented as the better long-term PH MT path, not the default training script |

### BART for minutes

| Choice | Why (project fact) | Alternatives considered / deferred |
|---|---|---|
| `facebook/bart-large-cnn` on **English only** | Strong abstractive EN summarization; PH speech is handled upstream | Summarizing raw Hiligaynon with BART — rejected (BART is EN-centric) |
| Topic chunking + coverage restore | Long board transcripts exceed one BART window | Single-shot whole-transcript BART — insufficient for multi-hour meetings |
| Extractive fallback (`ALLOW_LLM_FALLBACK`) | Meetings still get bullets if transformers/BART fails | Hard-fail with empty summary — rejected for UX |

---

## Performance, latency, and cost

**One gap (merged):** size / latency / hardware notes and accuracy / cost
metrics are the same missing content area. DeepSeek’s list adds **cost**;
keep that in this single section — do not track two separate “performance”
issues.

### What we can state from defaults (not a benchmark)

| Stage | Default | Hardware posture in code | Notes |
|---|---|---|---|
| Whisper live | faster-whisper `small` | CPU or CUDA via `WHISPER_DEVICE` / `COMPUTE_TYPE` | ~10s window / 5s hop; optional Tagalog RNN-T for lower live latency |
| Whisper final | `medium` or HF PH (`auto`) | Same device settings; chunked (~600s) for long meetings | Stronger than live; dominates stop-recording wait |
| NLLB PH→EN | `nllb-200-distilled-600M` | Loaded on demand in `llm.py` | Preferred PH→EN path |
| mBART | `mbart-large-50-many-to-many-mmt` | Loaded on demand | Fallback PH→EN; primary for non-EN |
| BART | `bart-large-cnn` | Pipeline load uses **CPU** (`device=-1`) | Topic-aware; extractive fallback if load/fail |

Rough **download / disk** class (public HF cards; not Smart Meeting measurements):
Whisper `small` ≪ `medium` ≪ PH-medium HF; NLLB distilled-600M and
mBART-large-50 and BART-large-cnn are each multi‑GB. First run downloads
models unless cached.

### What the team must supply (do not invent)

| Metric | Status |
|---|---|
| Hiligaynon WER on board audio (live vs final vs PLD fine-tune) | **TBD** — use `scripts/hiligaynon_asr/wer.py` on held-out labels |
| End-to-end latency (stop → refined transcript; summarize wall time) | **TBD** — `/api/health/transcription` only probes live ASR in dev |
| Cloud/GPU cost per meeting-hour | **TBD** — depends on host (CPU vs GPU) and whether models stay warm |
| Trade-off table (accuracy vs spend) for choosing `small`/`medium`/fine-tune | **TBD** — product decision after the measurements above |

Until those exist, this doc records **architecture and defaults**, not
claimed WER/% or dollar figures.

Operators: `GET /api/health` → `pipeline` includes size/hardware **hints**
derived from configured model ids (not live benchmarks).

---

## Whisper — speech recognition

**Role in this project:** Turn meeting audio into text. Everything else
(summary, translation, history search) depends on this transcript.

### What it does here

- **Live pass** during recording: fast captions over WebSocket windows
  (default faster-whisper `small`, ~10s windows / 5s hop).
- **Final pass** when you stop (or retranscribe): stronger full-file ASR
  (default faster-whisper `medium`, or Hugging Face Philippine models when
  `WHISPER_FINAL_BACKEND=auto`).
- Meetings are Hiligaynon-biased (`WHISPER_DEFAULT_LANGUAGE=hil`). Whisper has
  **no native `hil` token**, so decode uses **auto-detect** plus Hiligaynon
  prompts — it does **not** force Tagalog (`tl`).
- Final Hiligaynon candidate order: custom fine-tune →
  `rbcurzon/whisper-medium-ph` → faster-whisper `medium`.
- Optional: Tagalog-only live RNN-T; PLD fine-tune tooling for better Hiligaynon.

### Inputs / outputs

| Field | Detail |
|---|---|
| Input | 16 kHz mono PCM (live) or archived WAV (final) |
| Output | Plain transcript + timed segments → `meeting.final_transcript` |
| Trigger | WebSocket while recording; finalize on stop / upload / retranscribe |

**Settings:** `WHISPER_DEFAULT_LANGUAGE`, `WHISPER_FINAL_BACKEND`,
`WHISPER_LIVE_MODEL`, `WHISPER_FINAL_MODEL`, `WHISPER_HILIGAYNON_*`

**Key files:** `backend/app/services/transcription.py`, `asr.py`,
`ws/transcription.py`, `finalize.py`

**Docs:** [`FINE_TUNE_HILIGAYNON.md`](FINE_TUNE_HILIGAYNON.md), [`PLD.md`](PLD.md),
[`RNN_T_LIVE.md`](RNN_T_LIVE.md)

**Project in one line (Whisper’s view):** “I listen to the board meeting and
produce the authoritative transcript; live captions are temporary, the final
pass is what summaries and translations read.”

---

## BART — English meeting minutes

**Role in this project:** Condense **already-English** text with topic-aware
BART. ``source_kind="meeting"`` (default) uses board framing and Discussion /
Decisions / Action items. ``source_kind="general"`` uses a neutral
“Summarize the following.” frame and flat/topic bullets with **no** minutes
bucketing — so song/narrative content is not force-sorted into Action items.

### What it does here

- Default model: `facebook/bart-large-cnn` (`BART_MODEL`).
- Runs only on `POST /api/ai/summarize`, **after** the transcript has been
  translated to English (see mBART/NLLB below).
- Topic-aware: splits the English text by similarity, summarizes per topic,
  then restores coverage for idea units BART dropped.
- Falls back to extractive bullets if the generative path fails
  (`ALLOW_LLM_FALLBACK`).

### Inputs / outputs

| Field | Detail |
|---|---|
| Input | Normalized English text (preferred), not raw PH speech |
| Output | Formatted minutes string (`• …` or `1. …`) |
| Trigger | Explicit summarize API call (not automatic on stop; UI auto-runs after finalize) |

**Settings:** `BART_MODEL`, `BART_MAX_INPUT_TOKENS`, `BART_TOPIC_*`,
`ALLOW_LLM_FALLBACK`

**Key files:** `backend/app/services/llm.py`, `routers/ai.py`

**Project in one line (BART’s view):** “I never hear the audio — I only see
English text that mBART/NLLB already produced, and I turn that into short
board-meeting minutes.”

---

## mBART — translation (with NLLB for PH→EN)

**Role in this project:** Make the transcript usable across languages.
For summarization, that means **Philippine / mixed speech → English** first;
separately, translate the transcript into other UI languages on demand.

### What it does here

- Stock mBART: `facebook/mbart-large-50-many-to-many-mmt` (`MBART_MODEL`).
- Three-way router: EN passthrough / Tagalog→NLLB `tgl_Latn` / Hiligaynon→Google
  `hil` (NLLB `ceb_Latn` then mBART only as degraded fallback).
- Stock mBART Tagalog uses native **`tl_XX`** (not `id_ID` — see
  [`MBART_PH_AUDIT.md`](MBART_PH_AUDIT.md)). Hiligaynon has **no** mBART token.
- Optional `MBART_PH_FINE_TUNED_MODEL` (LoRA on `tl_XX`) for better Tagalog→EN.
- Non-English targets (es, fr, de, …) use mBART many-to-many.
- Sliding context windows with span-extraction safeguards for long transcripts.

### Inputs / outputs

| Field | Detail |
|---|---|
| Input | Full meeting transcript (any source language) |
| Output | Plain translated string |
| Trigger | Always as step 1 of summarize (→ English); also `POST /api/ai/translate` |

**Settings:** `MBART_MODEL`, `NLLB_MODEL`, `PH_TRANSLATE_BACKEND`,
`MBART_PH_FINE_TUNED_MODEL`

**Key files:** `backend/app/services/llm.py`, `languages.py`, `routers/ai.py`

**Docs:** [`FINE_TUNE_MBART_PH.md`](FINE_TUNE_MBART_PH.md),
[`MBART_PH_AUDIT.md`](MBART_PH_AUDIT.md), [`MT_TAG_BENCHMARK.md`](MT_TAG_BENCHMARK.md)

**Project in one line (mBART’s view):** “I bridge languages — I turn the Whisper
transcript into English so BART can write minutes, and I translate the same
transcript into other languages when the user asks.”

---

## How they fit together

| Stage | Model | User-visible result |
|---|---|---|
| During meeting | Whisper (live) | Live captions |
| Stop / refine | Whisper (final) | Refined transcript |
| Summarize | NLLB/mBART → BART | English meeting minutes |
| Translate | mBART (+ NLLB for EN) | Transcript in chosen language |

Whisper owns **accuracy of what was said**. mBART/NLLB owns **language access**.
BART owns **readable minutes** from English only.

Runtime: `GET /api/health` → `pipeline`.

See [`PRODUCT.md`](PRODUCT.md) for the full product overview and how each
claim maps to shipped vs roadmap features.

---

## Glossary

| Term | Meaning |
|---|---|
| **ASR** | Automatic Speech Recognition — the general task Whisper performs. |
| **NLLB** | No Language Left Behind — Meta’s multilingual translation model family, used here as the preferred PH→EN backend. |
| **RNN-T** | Recurrent Neural Network Transducer — a streaming speech-recognition architecture used here for the optional Tagalog-only live captioning path. |
| **PLD** | **Philippine Languages Database** (UP-DSP PLD; Guevara et al., SIGUL @ LREC-COLING 2024) — open speech corpus (~454h, 10 languages including ~41h Hiligaynon). Smart Meeting uses it as **training data** for Whisper fine-tuning, not as a runtime checkpoint. See [`PLD.md`](PLD.md). |
| **WER** | Word Error Rate — standard ASR accuracy metric (lower is better). Measure with `scripts/hiligaynon_asr/wer.py` against human transcripts. |
| **HF** | Hugging Face — model hub / `transformers` checkpoints used for PH Whisper, BART, mBART, and NLLB. |

Acronyms and jargon share **one** fix: this glossary. Do not track
“undefined acronyms” and “technical jargon” as separate open issues.

---

## Review follow-ups (de-duplicated)

### Closed (documentation / structure)

| Feedback | Disposition |
|---|---|
| Missing H1/H2/H3 / structural inconsistency | **Closed** — heading levels + Settings / Key files / Docs order standardized |
| Technical jargon / undefined acronyms | **Closed** — single [Glossary](#glossary) |
| Mixed notation (backslashes vs arrows) | **Dismissed** — source uses consistent arrows; not actionable |
| Vague “poor flow” / “buried concepts” without passages | **Parked** — need specific paragraphs before editing |
| “Too technical for PMs” + “too brief for developers” | **Merged** into [Who this is for](#who-this-is-for) |

### Open (needs team-supplied content or measurement)

| Feedback | Disposition |
|---|---|
| No model size / latency / hardware notes **and** performance metrics (accuracy, latency, **costs**) | **Merged into one gap** — see [Performance…](#performance-latency-and-cost); fill TBD rows with real numbers |
| Why models chosen / alternatives | **Addressed with project facts** above; expand only if product wants a longer ADR |
| Audience targeting | **Addressed** in [Who this is for](#who-this-is-for) |
| Table of contents | **Added** — useful now that the doc has exec summary + audience + rationale |

### Already applied in the product (not doc-only)

- `pipeline` on `/api/health` (roles + size/hardware hints)
- Meeting-room Whisper → mBART/NLLB → BART stage strip
- `NLLB_MODEL` in `.env.example`
