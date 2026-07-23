# Philippine Languages Database (PLD)

Smart Meeting targets Hiligaynon board meetings. The best open training source
for that is the **UP-DSP Philippine Languages Database (PLD)**
(Guevara et al., SIGUL @ LREC-COLING 2024).

| | |
|---|---|
| Paper | [ACL Anthology 2024.sigul-1.32](https://aclanthology.org/2024.sigul-1.32/) |
| Dataset | [UP-DSP-PLD on Mozilla Data Collective](https://mozilladatacollective.com/datasets/cmmxhw46c00tqnw07xyr94zjk) |
| Size | ~454 hours, 10 languages, 16 kHz WAV |
| Hiligaynon | ~41 hours, 91 speakers (read + spontaneous) |
| License | Research / CC-BY-NC-4.0 (check the download portal) |

PLD is a **corpus**, not a ready-made Whisper checkpoint. Use it to fine-tune,
then point Smart Meeting at your model.

## Languages in PLD

Filipino, English, Cebuano, Kapampangan, **Hiligaynon**, Ilokano, Bikolano,
Waray, Tausug, Pangasinense.

## Recommended path (Hiligaynon)

```bash
# 1) Download UP-DSP-PLD and unpack so you have something like:
#    data/PLD/HIL/<speaker_id>/*.wav + *.log

# 2) Import Hiligaynon sessions → JSONL for our trainer
python scripts/hiligaynon_asr/import_pld.py \
  --pld-root ./data/PLD \
  --language hil \
  --output ./hil-pld-train.jsonl

# Optional: hold out a slice for WER
python scripts/hiligaynon_asr/import_pld.py \
  --pld-root ./data/PLD --language hil \
  --output ./hil-pld-eval.jsonl --limit 500

# 3) Fine-tune Whisper (GPU recommended; no forced Tagalog token)
python scripts/hiligaynon_asr/finetune_whisper.py \
  --train-jsonl ./hil-pld-train.jsonl \
  --eval-jsonl ./hil-pld-eval.jsonl \
  --output-dir ./models/whisper-medium-pld-hil \
  --model-name openai/whisper-medium \
  --fp16

# 4) Plug into Smart Meeting
# .env
WHISPER_HILIGAYNON_FINE_TUNED_MODEL=/path/to/models/whisper-medium-pld-hil
WHISPER_FINAL_BACKEND=auto
WHISPER_HILIGAYNON_FINAL_LANGUAGE_MODE=auto

# Optional live CT2 export
./scripts/hiligaynon_asr/export_ct2.sh \
  ./models/whisper-medium-pld-hil \
  ./models/whisper-medium-pld-hil-ct2
# WHISPER_LIVE_HILIGAYNON_MODEL=/path/to/models/whisper-medium-pld-hil-ct2
```

## Other PLD languages

The same importer works for `ceb`, `ilo`, `war`, `bik`, `pam`, `pag`, `tsg`,
`fil` — useful if you later expand Smart Meeting beyond Hiligaynon/Tagalog.

```bash
python scripts/hiligaynon_asr/import_pld.py \
  --pld-root ./data/PLD --language ceb --output ./ceb-pld-train.jsonl
```

Optional community tooling: [`dka-speech`](https://pypi.org/project/dka-speech/)
also imports PLD session folders (`dka build … --preset pld`) into HF CSVs.
Our `import_pld.py` writes the JSONL format this repo’s trainer already expects.

## Runtime note

Until a PLD fine-tune is configured, Smart Meeting keeps using:

1. `WHISPER_HILIGAYNON_FINE_TUNED_MODEL` (empty by default)
2. `rbcurzon/whisper-medium-ph` (PH dialect HF fallback)
3. faster-whisper `medium`

Hiligaynon decode stays on **Whisper auto-detect** (never forced Tagalog `tl`).
