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
| Fine-tune Whisper on Hiligaynon audio + transcripts | **Full FT:** `finetune_whisper.py`. **LoRA (recommended):** `finetune_whisper_lora.py` + `merge_whisper_lora.py`. |
| LoRA on attention / FFN (`q_proj`, `v_proj`, `out_proj`, `fc1`, `fc2`) | Supported — default r=64, α=128; optional `--load-in-4bit` on CUDA. |
| Use ~tens–100+ hours when possible (PLD ~41h Hiligaynon) | Documented; prep via `prepare_whisper_pld.py`. Even a few clean hours helps. |
| faster-whisper / SpeechBrain fine-tunes | Final ASR uses HF transformers checkpoints. Live captions use a **CTranslate2** export (`export_ct2.sh`) with faster-whisper. |
| Improve accuracy beyond stock Whisper | Runtime: custom fine-tune → `rbcurzon/whisper-medium-ph` → faster-whisper, plus Hiligaynon prompt bias and auto-detect (no forced `tl`). Preferred training data: **UP-DSP PLD** — see [`docs/PLD.md`](PLD.md). |

## Scripts (train → evaluate → plug in)

### A) Recommended: LoRA fine-tune (cheaper VRAM)

Adapted from the Hiligaynon LoRA guide (PEFT on `q_proj`/`v_proj`/`out_proj`/`fc1`/`fc2`).
Does **not** force a Whisper `hil` language token (stock Whisper has none).

```bash
# 0) Preferred: clean + speaker-disjoint PLD splits
#    See docs/PLD.md
python3 scripts/hiligaynon_asr/prepare_whisper_pld.py \
  --pld-root ./data/PLD --language hil --out-dir ./data/pld_hiligaynon_clean

# 1) LoRA train (GPU recommended)
pip install "transformers>=4.40" datasets accelerate peft torch \
  librosa soundfile evaluate jiwer
# Optional 4-bit on CUDA: pip install bitsandbytes  →  add --load-in-4bit
python3 scripts/hiligaynon_asr/finetune_whisper_lora.py \
  --train-jsonl ./data/pld_hiligaynon_clean/train.jsonl \
  --eval-jsonl ./data/pld_hiligaynon_clean/dev.jsonl \
  --test-jsonl ./data/pld_hiligaynon_clean/test.jsonl \
  --output-dir ./models/whisper-medium-hil-lora \
  --model-name openai/whisper-medium \
  --lora-r 64 --lora-alpha 128 \
  --num-train-epochs 10 --fp16

# 2) Merge adapters → full transformers checkpoint
python3 scripts/hiligaynon_asr/merge_whisper_lora.py \
  --adapter-dir ./models/whisper-medium-hil-lora \
  --output-dir ./models/whisper-medium-hiligaynon

# 3) WER on held-out pairs (optional)
python3 scripts/hiligaynon_asr/wer.py --pair ./eval/pairs.jsonl

# 4) Optional: CT2 for live captions
./scripts/hiligaynon_asr/export_ct2.sh \
  ./models/whisper-medium-hiligaynon \
  ./models/whisper-medium-hiligaynon-ct2
```

### B) Full fine-tune (no LoRA)

```bash
# Or build JSONL from your own WAV+TXT pairs
python3 scripts/hiligaynon_asr/prepare_dataset.py \
  --input-dir ./hil-data --output ./hil-train.jsonl

python3 scripts/hiligaynon_asr/finetune_whisper.py \
  --train-jsonl ./data/pld_hiligaynon_clean/train.jsonl \
  --eval-jsonl ./data/pld_hiligaynon_clean/dev.jsonl \
  --output-dir ./models/whisper-medium-hiligaynon \
  --model-name openai/whisper-medium \
  --fp16
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
