# Fine-tuning Whisper for Hiligaynon

Stock Whisper has **no native Hiligaynon (`hil`) language token**. Smart Meeting
labels meetings as `hil` but decodes with the closest supported Philippine code
(`tl` / Tagalog) or auto-detect. That mismatch is the main source of Hiligaynon
transcription discrepancy.

**Fine-tuning is the most effective fix.** This repo does not train models
in-place (that needs your labeled audio). It **loads** a fine-tuned checkpoint
you produce externally.

## Why fine-tune?

Research on other low-resource languages (e.g. Basque with ~116h on
`whisper-medium`) shows large WER drops versus out-of-the-box Whisper. The same
approach applies to Hiligaynon: teach the model your dialect, domain vocabulary
(board meetings), and code-switching with Filipino / English.

## Recommended workflow (outside this repo)

1. **Collect data** — Hiligaynon (and mixed EN/FIL) audio with verified text.
   Aim for tens of hours when possible; even a few carefully cleaned hours helps.
2. **Format** — 16 kHz mono WAV/FLAC + matching transcripts (CSV / JSONL /
   Hugging Face `Dataset`).
3. **Fine-tune** — start from `openai/whisper-medium` (or `small` for CPU-only
   live use) with Hugging Face Transformers / SpeechBrain / OpenAI Whisper
   fine-tune scripts. Keep `task=transcribe`.
4. **Export**
   - **Final ASR (transformers):** save a normal HF Whisper folder or push a
     private/public Hub repo.
   - **Live ASR (faster-whisper):** convert the fine-tune to CTranslate2 with
     [`ct2-transformers-converter`](https://github.com/OpenNMT/CTranslate2) so
     low-latency captions can use it.
5. **Point Smart Meeting at the checkpoint** (see below). Restart the API.

## Configure Smart Meeting

In `backend/.env`:

```bash
# Prefer HF Hiligaynon candidates for final pass (default).
WHISPER_FINAL_BACKEND=auto

# 1) Your fine-tune — tried first (HF repo or local transformers folder)
WHISPER_HILIGAYNON_FINE_TUNED_MODEL=/path/to/your-hiligaynon-whisper
# or: WHISPER_HILIGAYNON_FINE_TUNED_MODEL=your-org/whisper-medium-hiligaynon

# 2) Philippine HF fallback if the custom fine-tune is empty / fails
WHISPER_HILIGAYNON_MODEL=rbcurzon/whisper-medium-ph

# Optional: CTranslate2 export for live captions
WHISPER_LIVE_HILIGAYNON_MODEL=/path/to/your-hiligaynon-whisper-ct2

# Meeting label stays hil; Whisper decode code stays tl (or auto for final).
WHISPER_DEFAULT_LANGUAGE=hil
WHISPER_DECODE_LANGUAGE=tl
WHISPER_FINAL_LANGUAGE_MODE=auto

# Optional prompt bias (keep short — long prompts can be echoed).
WHISPER_INITIAL_PROMPT=Board meeting discussion in Hiligaynon (Ilonggo), Filipino, and English.
```

**Candidate order for `hil` meetings:** custom fine-tune →
`rbcurzon/whisper-medium-ph` → faster-whisper `medium` (or your CT2 export).

Until you have your own fine-tune, leave `WHISPER_HILIGAYNON_FINE_TUNED_MODEL`
empty; the Philippine checkpoint is used when transformers/torch are installed
(`pip install -r requirements-ml.txt`).

## Verify

```bash
curl -s http://127.0.0.1:8000/api/health | jq '{
  whisper_final_backend,
  whisper_final_backend_resolved_hil,
  whisper_hiligaynon_fine_tuned_model,
  whisper_hiligaynon_model,
  whisper_hiligaynon_hf_candidates,
  whisper_live_hiligaynon_model
}'
```

Then re-transcribe a known Hiligaynon meeting from History and compare WER
against a human transcript.

## Practical tips

- Prefer **medium** for accuracy; use **small** CT2 for live if CPU-bound.
- Include **code-switched** EN/FIL/Hiligaynon clips — board meetings are mixed.
- Keep evaluation clips held out from training.
- If the fine-tune hallucinates, Smart Meeting falls back to faster-whisper
  automatically for that pass.
