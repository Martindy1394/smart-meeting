# Smart Meeting — model summaries

Pipeline: **audio → Whisper (transcript) → mBART/NLLB (English) → BART (minutes)**,
with optional mBART translation to other languages on demand.

```
Live PCM ──► Whisper (live captions)
Full WAV ──► Whisper (final transcript)
Transcript ──► NLLB / mBART ──► English ──► BART ──► meeting minutes
Transcript ──► mBART ──► other languages (on demand)
```

---

## Whisper — speech recognition

**Role in this project:** Turn meeting audio into text. Everything else (summary,
translation, history search) depends on this transcript.

**What it does here**

- **Live pass** during recording: fast captions over WebSocket windows
  (default faster-whisper `small`, ~10s windows / 5s hop).
- **Final pass** when you stop (or retranscribe): stronger full-file ASR
  (default faster-whisper `medium`, or Hugging Face Philippine models when
  `WHISPER_FINAL_BACKEND=auto`).
- Meetings are Hiligaynon-biased (`WHISPER_DEFAULT_LANGUAGE=hil`). Whisper has
  **no native `hil` token**, so decode uses **auto-detect** plus Hiligaynon
  prompts — it does **not** force Tagalog `tl`.
- Final Hiligaynon candidate order: custom fine-tune →
  `rbcurzon/whisper-medium-ph` → faster-whisper `medium`.
- Optional: Tagalog-only live RNN-T; PLD fine-tune tooling for better Hiligaynon.

**Inputs / outputs**

| | |
|---|---|
| Input | 16 kHz mono PCM (live) or archived WAV (final) |
| Output | Plain transcript + timed segments → `meeting.final_transcript` |
| Trigger | WebSocket while recording; finalize on stop / upload / retranscribe |

**Key files:** `backend/app/services/transcription.py`, `asr.py`,
`ws/transcription.py`, `finalize.py`  
**Docs:** [`FINE_TUNE_HILIGAYNON.md`](FINE_TUNE_HILIGAYNON.md), [`PLD.md`](PLD.md),
[`RNN_T_LIVE.md`](RNN_T_LIVE.md)

**Project in one line (Whisper’s view):** “I listen to the board meeting and
produce the authoritative transcript; live captions are temporary, the final
pass is what summaries and translations read.”

---

## BART — English meeting minutes

**Role in this project:** Condense the **English** transcript into structured
meeting minutes (Discussion / Decisions / Action items), as bullets or numbered
items.

**What it does here**

- Default model: `facebook/bart-large-cnn` (`BART_MODEL`).
- Runs only on `POST /api/ai/summarize`, **after** the transcript has been
  translated to English (see mBART/NLLB below).
- Topic-aware: splits the English text by similarity, summarizes per topic,
  then restores coverage for idea units BART dropped.
- Falls back to extractive bullets if the generative path fails
  (`ALLOW_LLM_FALLBACK`).

**Inputs / outputs**

| | |
|---|---|
| Input | Normalized English text (preferred), not raw PH speech |
| Output | Formatted minutes string (`• …` or `1. …`) |
| Trigger | Explicit summarize API call (not automatic on stop) |

**Key files:** `backend/app/services/llm.py`, `routers/ai.py`  
**Settings:** `BART_MODEL`, `BART_MAX_INPUT_TOKENS`, `BART_TOPIC_*`

**Project in one line (BART’s view):** “I never hear the audio — I only see
English text that mBART/NLLB already produced, and I turn that into short
board-meeting minutes.”

---

## mBART — translation (with NLLB for PH→EN)

**Role in this project:** Make the transcript usable across languages.
For summarization, that means **Philippine / mixed speech → English** first;
separately, translate the transcript into other UI languages on demand.

**What it does here**

- Stock mBART: `facebook/mbart-large-50-many-to-many-mmt` (`MBART_MODEL`).
- PH→EN default path (`PH_TRANSLATE_BACKEND=auto`): prefer
  `facebook/nllb-200-distilled-600M` (Hiligaynon via `ceb_Latn`, Tagalog via
  `tgl_Latn`), fall back to mBART (`id_ID` for hil/tl when no PH fine-tune).
- Optional `MBART_PH_FINE_TUNED_MODEL` for better Philippine → English.
- Non-English targets (es, fr, de, …, hil, tl) use mBART many-to-many.
- Sliding context windows so long board transcripts stay coherent.

**Inputs / outputs**

| | |
|---|---|
| Input | Full meeting transcript (any source language) |
| Output | Plain translated string |
| Trigger | Always as step 1 of summarize (→ English); also `POST /api/ai/translate` |

**Key files:** `backend/app/services/llm.py`, `languages.py`, `routers/ai.py`  
**Docs:** [`FINE_TUNE_MBART_PH.md`](FINE_TUNE_MBART_PH.md)

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
