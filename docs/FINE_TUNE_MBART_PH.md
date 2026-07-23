# Fine-tune mBART for Tagalog / Hiligaynon → English

Smart Meeting’s PH→English path defaults to **NLLB** (better Tagalog codes).
This guide fine-tunes **mBART-50** with LoRA so you can optionally prefer a
custom Philippine checkpoint.

## Data

| Source | Role |
|---|---|
| OPUS Tatoeba `en-tl` | Real Tagalog↔English sentences (~8.7k) |
| `scripts/ph_mt/seed_hiligaynon_en.jsonl` | Curated Hiligaynon meeting phrases |
| [pinoy-dictionary-scraper](https://github.com/luisligunas/pinoy-dictionary-scraper) | Optional short word→gloss pairs (capped) |

Hiligaynon has **no mBART language token**, so training uses `tl_XX` → `en_XX`
as a proxy for both Tagalog and Hiligaynon.

## Train

```bash
# 1) Download Tatoeba (example)
curl -L -o /tmp/en-tl.zip \
  https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/en-tl.txt.zip
unzip -o /tmp/en-tl.zip -d /tmp/opus-tl

# 2) Optional dictionaries
mkdir -p scripts/ph_mt/data
curl -L -o scripts/ph_mt/data/tagalog_dictionary.json \
  https://raw.githubusercontent.com/luisligunas/pinoy-dictionary-scraper/main/Scraped%20Data/Dictionaries/tagalog_dictionary.json
curl -L -o scripts/ph_mt/data/hiligaynon_dictionary.json \
  https://raw.githubusercontent.com/luisligunas/pinoy-dictionary-scraper/main/Scraped%20Data/Dictionaries/hiligaynon_dictionary.json

# 3) Build JSONL
python scripts/ph_mt/prepare_mbart_dataset.py \
  --tatoeba-dir /tmp/opus-tl \
  --dictionary-dir scripts/ph_mt/data \
  --hil-seed scripts/ph_mt/seed_hiligaynon_en.jsonl \
  --output-dir scripts/ph_mt/prepared

# 4) LoRA fine-tune (GPU strongly recommended)
pip install "transformers>=4.40" datasets accelerate peft sentencepiece protobuf torch
python scripts/ph_mt/finetune_mbart.py \
  --train-jsonl scripts/ph_mt/prepared/train.jsonl \
  --eval-jsonl scripts/ph_mt/prepared/eval.jsonl \
  --output-dir models/mbart-ph-en-lora \
  --num-train-epochs 1 \
  --fp16

# 5) Merge adapters into a full checkpoint
python scripts/ph_mt/merge_lora.py \
  --adapter-dir models/mbart-ph-en-lora \
  --output-dir models/mbart-ph-en-merged
```

## Configure Smart Meeting

```bash
# backend/.env
MBART_PH_FINE_TUNED_MODEL=/absolute/path/to/models/mbart-ph-en-merged
# Prefer the fine-tuned mBART for Philippine → English (NLLB remains fallback)
PH_TRANSLATE_BACKEND=mbart
```

`PH_TRANSLATE_BACKEND` values:

- `auto` — use fine-tuned mBART first when `MBART_PH_FINE_TUNED_MODEL` is set, else NLLB
- `mbart` — Philippine path uses mBART (`tl_XX` when fine-tuned)
- `nllb` — keep the current NLLB-first behavior

## Notes

- Dictionary glosses are a **weak** signal; do not raise `--dict-limit-per-lang` too high.
- Prefer more **meeting-domain** Hiligaynon/Tagalog sentence pairs for real gains.
- Full fine-tune without LoRA (`--no-lora`) needs a GPU with ample VRAM.
- A short CPU LoRA smoke run (`--max-steps 400`) validates the pipeline but is **not**
  enough for production quality — keep `PH_TRANSLATE_BACKEND=nllb` (or `auto` without
  a strong checkpoint) until you train ≥1–3 epochs on GPU.
- NLLB fine-tuning is still the better long-term PH MT path; this pipeline exists
  for teams that want an mBART checkpoint specifically.

## Local artifact layout (gitignored)

```
models/mbart-ph-en-lora/     # LoRA adapters from finetune_mbart.py
models/mbart-ph-en-merged/   # merge_lora.py output — point MBART_PH_FINE_TUNED_MODEL here
scripts/ph_mt/prepared/      # train.jsonl / eval.jsonl from prepare_mbart_dataset.py
scripts/ph_mt/data/          # optional pinoy-dictionary JSON downloads
```
