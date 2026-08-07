# Whisper ASR — why Tagalog & Hiligaynon words are inaccurate

**Scope:** Whisper ASR only (`backend/app/services/transcription.py`, `asr.py`, config).  
**Verified:** 2026-08-07 with runtime probes on the live codebase.

This is a root-cause scan, not a training guide. The pipeline *avoids* the old
mistake of forcing Hiligaynon as Tagalog (`tl`), but several layers still prevent
accurate Tagalog / Hiligaynon word forms.

---

## Bottom line

| Language | Main blockers |
|---|---|
| **Hiligaynon** | (1) Post-process **deletes real words** `gid` / `amo`. (2) Final scoring often **throws away PH-HF** for stock faster-whisper when detect=`tl`. (3) **No Hiligaynon fine-tune** loaded; Whisper has **no `hil` token**. (4) HF hil path uses **auto-detect without prompt**. |
| **Tagalog** | (1) Product default `auto → hil` **never selects** Tagalog HF / forced-`tl`. (2) When `tl` is used, primary model is **small + unstable**. (3) Same FW-over-HF preference can discard Tagalog HF. |

---

## P0 — Blockers (code)

### 1. Prompt-echo strip deletes Hiligaynon words `gid` and `amo`

**Where:** `transcription.py` → `_strip_initial_prompt_echo`  
**Prompt:** `WHISPER_HILIGAYNON_INITIAL_PROMPT` includes  
`Wala gid, indi, sang, kag, amo.`

`keep` allowlists `sang`, `kag`, `indi`, `wala` — but **not** `gid` or `amo`.
Those tokens are in the prompt, so they enter `drop` and are removed from real
transcripts (live sanitize, final sanitize, and quality scoring).

**Runtime proof:**

```
BEFORE: Wala gid kami sang budget subong. Amo ina ang plano.
AFTER : Wala kami sang budget subong. ina ang plano
DROPPED: amo, gid
```

Irony: `_VISAYAN_MARKERS` *boosts* `gid` for scoring, but strip removes it first.

---

### 2. Final selection prefers stock faster-whisper when FW detects `tl`/`en`

**Where:** `transcription.py` ~2103–2144

Even when HF already has a **higher** quality score, if FW language ∈
`{en, tl, …}` and `fw_score ≥ 8` and `hf_score < fw_score * 1.8`, code forces
`prefer_fw = True`.

**Simulation:** HF=19.4, FW=14.0, `fw_lang=tl` → **prefer FW** (discards better HF).

Hiligaynon audio often auto-detects as `tl` (no `hil` token), so this rule
systematically demotes `rbcurzon/whisper-medium-ph` in favor of stock
`faster-whisper:medium`, which maps Ilonggo into Tagalog/English-like words.

---

### 3. Product path is always `auto → hil` — Tagalog models unused

**Where:**

- `whisper_default_language = "hil"`
- `effective_asr_language("auto")` → `hil`
- UI saves meeting language as `auto` (no spoken-language picker)
- `philippine_hf_candidates("auto"|"hil")` → only PH-medium  
  **not** `LWobole/whisper-small-tagalog`

**Runtime:**

```
auto candidates: ['rbcurzon/whisper-medium-ph']
tl candidates:   ['LWobole/whisper-small-tagalog', 'rbcurzon/whisper-medium-ph']
```

Forced-`tl`, Tagalog prompt, and Tagalog HF are effectively dead on the default
product path. Tagalog speech is decoded as Hiligaynon-biased auto + PH-medium /
stock FW.

Session lock on `auto` also remaps Whisper-detected `tl` → `hil`, so a Tagalog
meeting never migrates onto the Tagalog path mid-session.

---

## P1 — High (model / config)

### 4. Empty fine-tune slots + no Whisper `hil` token

| Setting | Default |
|---|---|
| `WHISPER_HILIGAYNON_FINE_TUNED_MODEL` | empty |
| `WHISPER_TAGALOG_FINE_TUNED_MODEL` | empty |
| `WHISPER_LIVE_*` CT2 fine-tunes | empty |

`whisper_language_arg("hil")` → `None` (auto-detect). Correct (cannot force
`hil`), but without a PLD/LoRA checkpoint accuracy stays limited.

LoRA train/merge scripts exist (`scripts/hiligaynon_asr/finetune_whisper_lora.py`)
but nothing is wired into env yet.

### 5. Live captions = stock `small` + CPU `int8`

No PH/Tagalog CT2 live models → weak live words for both languages. Finalize can
keep live text when final is short.

### 6. Tagalog HF primary is `whisper-small` (unstable)

When language is explicitly `tl`, first candidate is
`LWobole/whisper-small-tagalog`. Accuracy review: WER 0.11–0.33 on the same clip;
occasional wrong-script tokens (`cái`, CJK).

### 7. Hiligaynon HF decode: no prompt + auto-detect → often `tl`

HF Hiligaynon path skips `prompt_ids` (echo avoidance) and sets
`forced_decoder_ids=None`. Lexical bias is left to FW only — then strip (#1) and
prefer-FW (#2) can still destroy or discard the better text.

---

## P2 — Medium amplifiers

| # | Issue | Effect |
|---|---|---|
| 8 | Tagalog `prefer_forced` → force `tl` | Right for pure TL; weak on EN+TL code-switch inside HF |
| 9 | Inner HF rank = words × coverage | High-coverage garble can win an attempt |
| 10 | Live vs final asymmetry | Live never sees PH-medium; words jump between caption and final |
| 11 | Clear `forced_decoder_ids` | Avoids transformers conflicts; also clears checkpoint language prior |

---

## What is *not* broken

- Hiligaynon is **not** forced to `tl` in decode helpers (intentional fix).
- `fil` → Whisper `tl`; `hil` → `None` (invalid codes not forwarded).
- RNN-T live is Tagalog-only and skipped for Hiligaynon (avoids TL lexicon on Ilonggo).

---

## Fix status (implemented 2026-08-07)

1. **Done:** `gid` / `amo` added to `_strip_initial_prompt_echo` `keep`.
2. **Done:** FW-over-HF `tl` override only when FW **outscores** HF (no more
   discard of higher HF via `hf < fw*1.8`).
3. **Done:** `auto` / hil HF candidate lists include Tagalog HF as scored secondary.
4. **Still needed:** Train/merge Hiligaynon LoRA → `WHISPER_HILIGAYNON_FINE_TUNED_MODEL`.
5. **Still needed:** Stronger Tagalog checkpoint when available; optional CT2 live.

---

## Related artifacts

- Multi-trial measurements: `data/model_accuracy_review/ASR_REVIEW.md`
- Fine-tune path: `docs/FINE_TUNE_HILIGAYNON.md`, `docs/PLD.md`
- Language forcing history: `docs/HILIGAYNON_LANGUAGE_FORCING.md`
