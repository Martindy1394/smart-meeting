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

Prefer **`py -3`** on Windows, or **`python3`** on Linux/macOS.

### Windows (`M:\MSCS\PLD`)

Open PowerShell in the Smart Meeting repo, then:

```powershell
# 1) Confirm Hiligaynon folder is visible under your PLD path
py -3 scripts/hiligaynon_asr/import_pld.py --pld-root "M:\MSCS\PLD" --inspect

# 2) Import Hiligaynon → JSONL
py -3 scripts/hiligaynon_asr/import_pld.py `
  --pld-root "M:\MSCS\PLD" `
  --language hil `
  --output ".\hil-pld-train.jsonl"
```

If inspect shows `HIL` directly at `M:\MSCS\PLD\HIL`, you can also do:

```powershell
py -3 scripts/hiligaynon_asr/import_pld.py `
  --pld-lang-dir "M:\MSCS\PLD\HIL" `
  --output ".\hil-pld-train.jsonl"
```

Wanted layout:

```text
M:\MSCS\PLD\HIL\<speaker_id>\*.wav
M:\MSCS\PLD\HIL\<speaker_id>\*.log
```

(Folder may be named `Hiligaynon` instead of `HIL` — the importer accepts both.)

### Linux / macOS

```bash
python3 scripts/hiligaynon_asr/import_pld.py --pld-root ./data --inspect

python3 scripts/hiligaynon_asr/import_pld.py \
  --pld-root ./data/PLD \
  --language hil \
  --output ./hil-pld-train.jsonl
```

### Fine-tune and plug in

```bash
# Optional eval slice
py -3 scripts/hiligaynon_asr/import_pld.py --pld-root "M:\MSCS\PLD" --language hil --output .\hil-pld-eval.jsonl --limit 500

py -3 scripts/hiligaynon_asr/finetune_whisper.py `
  --train-jsonl .\hil-pld-train.jsonl `
  --eval-jsonl .\hil-pld-eval.jsonl `
  --output-dir .\models\whisper-medium-pld-hil `
  --model-name openai/whisper-medium `
  --fp16

# backend/.env
WHISPER_HILIGAYNON_FINE_TUNED_MODEL=M:/MSCS/smart-meeting/models/whisper-medium-pld-hil
WHISPER_FINAL_BACKEND=auto
WHISPER_HILIGAYNON_FINAL_LANGUAGE_MODE=auto
```

## Other PLD languages

```powershell
py -3 scripts/hiligaynon_asr/import_pld.py --pld-root "M:\MSCS\PLD" --language ceb --output .\ceb-pld-train.jsonl
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
