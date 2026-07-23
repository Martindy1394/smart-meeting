# Fine-tuning Whisper for Hiligaynon

> **Tagalog note:** Tagalog/Filipino meetings (`tl`) already use Whisper’s native
> `tl` token. Runtime order is custom fine-tune → `WHISPER_TAGALOG_MODEL`
> (default `LWobole/whisper-small-tagalog`) → `rbcurzon/whisper-medium-ph` →
> faster-whisper, with `WHISPER_TAGALOG_FINAL_LANGUAGE_MODE=prefer_forced`.
> You can reuse the same training scripts below with Tagalog audio/transcripts
> and set `WHISPER_TAGALOG_FINE_TUNED_MODEL` instead.

Stock Whisper has **no native Hiligaynon (`hil`) language token**. Smart Meeting
labels meetings as `hil` and uses **Whisper auto-detect** with a Hiligaynon
prompt and PH dialect models — it does **not** force Tagalog (`tl`) decode for
Ilonggo speech (that mismatch was a major source of bad transcripts).

**Fine-tuning is the most effective fix** (same approach as low-resource
languages such as Basque on `whisper-medium`). This repo:

1. **Loads** a fine-tuned checkpoint at runtime (final + optional live CT2).
2. Ships **training / eval / export scripts** so you can produce that checkpoint
   from your own labeled Hiligaynon audio.

## Suggestion vs what Smart Meeting does

| Suggestion | Status in this project |
|---|---|
| Fine-tune Whisper on Hiligaynon audio + transcripts | Supported via `scripts/hiligaynon_asr/finetune_whisper.py` (HF Transformers). SpeechBrain is also fine — export a transformers-compatible folder. |
| Use ~tens–100+ hours when possible (Basque ~116h → ~8.7% WER) | Documented; you supply the dataset. Even a few clean hours helps. |
| faster-whisper / SpeechBrain fine-tunes | Final ASR uses HF transformers checkpoints. Live captions use a **CTranslate2** export (`export_ct2.sh`) with faster-whisper. |
| Improve accuracy beyond stock Whisper | Runtime path: custom fine-tune → `rbcurzon/whisper-medium-ph` → faster-whisper, plus prompt bias and auto/`tl` coverage retries. |

## Scripts (train → evaluate → plug in)

```bash
# 1) Build JSONL from WAV+TXT pairs (or CSV)
python scripts/hiligaynon_asr/prepare_dataset.py \
  --input-dir ./hil-data --output ./hil-train.jsonl

# 2) Fine-tune whisper-medium (GPU recommended)
python scripts/hiligaynon_asr/finetune_whisper.py \
  --train-jsonl ./hil-train.jsonl \
  --output-dir ./models/whisper-medium-hiligaynon \
  --model-name openai/whisper-medium \
  --fp16

# 3) Measure WER on a held-out set
python scripts/hiligaynon_asr/wer.py \
  --reference ./eval/ref.txt --hypothesis ./eval/hyp.txt

# 4) Optional: export CT2 for live captions
./scripts/hiligaynon_asr/export_ct2.sh \
  ./models/whisper-medium-hiligaynon \
  ./models/whisper-medium-hiligaynon-ct2
```

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

# Meeting label stays hil; Whisper uses auto-detect (never forced Tagalog).
WHISPER_DEFAULT_LANGUAGE=hil
WHISPER_HILIGAYNON_FINAL_LANGUAGE_MODE=auto
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
against a human transcript with `scripts/hiligaynon_asr/wer.py`.

## Practical tips

- Prefer **medium** for accuracy; use **small** CT2 for live if CPU-bound.
- Include **code-switched** EN/FIL/Hiligaynon clips — board meetings are mixed.
- Keep evaluation clips held out from training.
- If the fine-tune hallucinates, Smart Meeting falls back to the next HF
  candidate, then faster-whisper, automatically.
