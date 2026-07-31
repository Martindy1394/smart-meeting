# Fine-tune mBART for Tagalog / Hiligaynon → English

Smart Meeting’s PH→English path defaults to **NLLB** (better Tagalog codes) and
**Google** for Hiligaynon. This guide fine-tunes **mBART-50** with LoRA so you
can optionally point `MBART_PH_FINE_TUNED_MODEL` at a custom checkpoint.

**Tagalog and Hiligaynon are two different jobs** — do not treat them as one
fine-tune without reading both sections.

| Language | Token | Training mode |
|---|---|---|
| Tagalog | existing `tl_XX` | standard LoRA (`q_proj`/`v_proj`, r=16, α=32) |
| Hiligaynon | **new** `hil_XX` | add token → resize → copy `tl_XX` embedding → full trainable embed + LoRA |

## Data

### Tagalog (`--lang tl`)

| Source | Role |
|---|---|
| OPUS Tatoeba `en-tl` | Real Tagalog↔English sentences (~8.7k) |
| OPUS-100 `en-tl` (optional `--opus100-max`) | Broader OPUS-family bitext |
| FLORES-200 `tgl_Latn` | Held-out chrF/BLEU by default (`--download-flores`) |
| `scripts/ph_mt/seed_tagalog_en.jsonl` | Domain meeting + lyric pairs |
| `scripts/ph_mt/fixtures/tagalog_en_sample.jsonl` | **Always held out** for domain go/no-go |
| pinoy-dictionary glosses (capped) | Weak optional signal |

### Hiligaynon (`--lang hil`)

| Source | Role |
|---|---|
| `scripts/ph_mt/seed_hiligaynon_en.jsonl` | Curated meeting phrases (primary seed) |
| `scripts/ph_mt/fixtures/hiligaynon_en_sample.jsonl` | **Always held out** for domain go/no-go |
| SEACrowd / OPUS Bible / JW300 exports | Pass via `--extra-jsonl` when you have them |
| pinoy-dictionary Hiligaynon glosses | Weak optional signal |

Expect **~10× less** usable Hiligaynon bitext than Tagalog. Budget quality
accordingly; keep Google Translate as production primary.

## Train — Tagalog

```bash
# 1) Build JSONL (downloads Tatoeba + FLORES held-out)
python scripts/ph_mt/prepare_mbart_dataset.py --lang tl \
  --download-opus --download-flores \
  --domain-seed scripts/ph_mt/seed_tagalog_en.jsonl \
  --output-dir scripts/ph_mt/prepared/tl

# 2) LoRA fine-tune (GPU strongly recommended)
pip install "transformers>=4.40" datasets accelerate peft sentencepiece \
  protobuf torch sacrebleu
python scripts/ph_mt/finetune_mbart.py --lang tl \
  --train-jsonl scripts/ph_mt/prepared/tl/train.jsonl \
  --eval-jsonl scripts/ph_mt/prepared/tl/eval.jsonl \
  --output-dir models/mbart-tl-en-lora \
  --lora-r 16 --lora-alpha 32 \
  --num-train-epochs 1 --fp16

# 3) Merge adapters
python scripts/ph_mt/merge_lora.py \
  --adapter-dir models/mbart-tl-en-lora \
  --output-dir models/mbart-tl-en-merged

# 4) Evaluate vs stock id_ID / tl_XX baselines (required before deploy)
python scripts/ph_mt/evaluate_mbart_checkpoint.py \
  --checkpoint models/mbart-tl-en-merged \
  --lang tl \
  --domain-jsonl scripts/ph_mt/fixtures/tagalog_en_sample.jsonl \
  --eval-jsonl scripts/ph_mt/prepared/tl/eval.jsonl \
  --out data/mbart_ph_ft/tl_eval_report.json
```

## Train — Hiligaynon (vocab extension)

```bash
python scripts/ph_mt/prepare_mbart_dataset.py --lang hil \
  --hil-seed scripts/ph_mt/seed_hiligaynon_en.jsonl \
  --output-dir scripts/ph_mt/prepared/hil

# Adds hil_XX, copies tl_XX embedding, modules_to_save=embed_tokens/lm_head,
# higher --embed-lr for the new row, LoRA on q_proj/v_proj for the rest.
python scripts/ph_mt/finetune_mbart.py --lang hil \
  --train-jsonl scripts/ph_mt/prepared/hil/train.jsonl \
  --eval-jsonl scripts/ph_mt/prepared/hil/eval.jsonl \
  --output-dir models/mbart-hil-en-lora \
  --lora-r 16 --lora-alpha 32 \
  --learning-rate 5e-5 --embed-lr 5e-4 \
  --num-train-epochs 3 --fp16

python scripts/ph_mt/merge_lora.py \
  --adapter-dir models/mbart-hil-en-lora \
  --output-dir models/mbart-hil-en-merged

python scripts/ph_mt/evaluate_mbart_checkpoint.py \
  --checkpoint models/mbart-hil-en-merged \
  --lang hil \
  --domain-jsonl scripts/ph_mt/fixtures/hiligaynon_en_sample.jsonl \
  --out data/mbart_ph_ft/hil_eval_report.json
```

Merged Hiligaynon checkpoints write `finetune_meta.json` with
`"has_hil_xx": true`. Smart Meeting’s `mbart_code("hil")` then returns
`hil_XX` automatically.

## Go / no-go (required)

Do **not** wire a checkpoint into production until
`evaluate_mbart_checkpoint.py` prints `GO`.

| Check | Tagalog | Hiligaynon |
|---|---|---|
| Domain token-F1 vs stock `id_ID` (~0.34) | must beat by ≥0.02 | must beat by ≥0.05 |
| Domain token-F1 vs stock `tl_XX` (~0.43) | must beat by ≥0.02 | must beat tl mistag of hil |
| BLEU / chrF | reported on domain + held-out | reported (expect low) |
| `hil_XX` in tokenizer | n/a | **required** |

Historical stock baselines: `docs/MBART_PH_AUDIT.md` / `data/mt_tag_benchmark/report.json`.

CPU smoke runs (`--max-steps 50`) validate the pipeline only — treat as **NO-GO**.

## Configure Smart Meeting (only after GO)

```bash
# backend/.env — Tagalog FT example
MBART_PH_FINE_TUNED_MODEL=/absolute/path/to/models/mbart-tl-en-merged
PH_TRANSLATE_BACKEND=mbart   # or auto
```

`PH_TRANSLATE_BACKEND`:

- `auto` — fine-tuned mBART first when `MBART_PH_FINE_TUNED_MODEL` is set, else NLLB
- `mbart` — Tagalog path uses mBART (`tl_XX`)
- `nllb` — NLLB-first

Hiligaynon routing stays **Google → NLLB → mBART last**, even with a
`hil_XX` fine-tune. The fine-tune only improves that last-resort path.

## Artifact layout (gitignored weights)

```
models/mbart-tl-en-lora/       # Tagalog LoRA adapters
models/mbart-tl-en-merged/     # merged — point MBART_PH_FINE_TUNED_MODEL here
models/mbart-hil-en-lora/      # Hiligaynon LoRA + shared_embeddings.pt
models/mbart-hil-en-merged/    # has_hil_xx=true in finetune_meta.json
scripts/ph_mt/prepared/{tl,hil}/
data/mbart_ph_ft/              # eval reports + TRAIN_REPORT.md (committed)
```

## Latest smoke-run verdict

See [`data/mbart_ph_ft/TRAIN_REPORT.md`](../data/mbart_ph_ft/TRAIN_REPORT.md).
CPU smoke (2026-07-31): Tagalog and Hiligaynon both show **metric improvement**
over stock baselines but are **PROVISIONAL / NO-GO for production** until a
GPU multi-epoch retrain clears the absolute quality floors.
