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

### Important: where you run the command

`M:\MSCS\PLD` exists on your **Windows PC**. The Cursor cloud agent terminal is
**Linux** (`/workspace/…`) and cannot see the `M:` drive — that is why you saw
`inspect root: /workspace/M:/MSCS/PLD` / path does not exist.

- To use files on `M:\MSCS\PLD`, open a **local** terminal in your repo clone on
  Windows (Git Bash or PowerShell) and run the commands there.
- Or copy `M:\MSCS\PLD\HIL` into the cloud workspace as `/workspace/data/PLD/HIL`
  and then use `--pld-root /workspace/data/PLD`.

### Windows local (`M:\MSCS\PLD`)

**If your terminal is bash / Git Bash** (prompt looks like `user@pc MINGW64`), use
`python3` and put the whole command on **one line** (do not use PowerShell `` ` ``):

```bash
python3 scripts/hiligaynon_asr/import_pld.py --pld-root "M:/MSCS/PLD" --inspect

python3 scripts/hiligaynon_asr/import_pld.py --pld-root "M:/MSCS/PLD" --language hil --output ./hil-pld-train.jsonl
```

Git Bash path form also works: `--pld-root "/m/MSCS/PLD"`.

**If your terminal is PowerShell**, use `py -3` and backticks for line breaks:

```powershell
py -3 scripts/hiligaynon_asr/import_pld.py --pld-root "M:\MSCS\PLD" --inspect

py -3 scripts/hiligaynon_asr/import_pld.py `
  --pld-root "M:\MSCS\PLD" `
  --language hil `
  --output ".\hil-pld-train.jsonl"
```

If inspect shows `HIL` at `M:\MSCS\PLD\HIL`:

```bash
python3 scripts/hiligaynon_asr/import_pld.py --pld-lang-dir "M:/MSCS/PLD/HIL" --output ./hil-pld-train.jsonl
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

### Clean + split for Whisper (recommended)

Same filtering ideas as OmniVoice Filipino PLD prep (drop spontaneous /
digit-paren noise, keep 1–15s clips, speaker-disjoint train/dev/test), adapted
to Smart Meeting's Whisper JSONL trainer:

```bash
python3 scripts/hiligaynon_asr/prepare_whisper_pld.py \
  --pld-root ./data/PLD \
  --language hil \
  --out-dir ./data/pld_hiligaynon_clean

# Optional standalone package (hardlink WAVs to save disk):
# python3 scripts/hiligaynon_asr/prepare_whisper_pld.py \
#   --pld-root ./data/PLD --language hil \
#   --out-dir ./data/pld_hiligaynon_clean \
#   --package-dir ./data/pld_hiligaynon_clean_pkg --package-mode hardlink
```

Windows / Git Bash (one line):

```bash
python3 scripts/hiligaynon_asr/prepare_whisper_pld.py --pld-root "M:/MSCS/PLD" --language hil --out-dir ./data/pld_hiligaynon_clean
```

### Fine-tune and plug in

Use the cleaned speaker-disjoint splits from `prepare_whisper_pld.py`:

```bash
python3 scripts/hiligaynon_asr/finetune_whisper.py \
  --train-jsonl ./data/pld_hiligaynon_clean/train.jsonl \
  --eval-jsonl ./data/pld_hiligaynon_clean/dev.jsonl \
  --output-dir ./models/whisper-medium-pld-hil \
  --model-name openai/whisper-medium \
  --fp16

# backend/.env
WHISPER_HILIGAYNON_FINE_TUNED_MODEL=M:/MSCS/smart-meeting/models/whisper-medium-pld-hil
WHISPER_FINAL_BACKEND=auto
WHISPER_HILIGAYNON_FINAL_LANGUAGE_MODE=auto
```

Windows (PowerShell):

```powershell
py -3 scripts/hiligaynon_asr/finetune_whisper.py `
  --train-jsonl .\data\pld_hiligaynon_clean\train.jsonl `
  --eval-jsonl .\data\pld_hiligaynon_clean\dev.jsonl `
  --output-dir .\models\whisper-medium-pld-hil `
  --model-name openai/whisper-medium `
  --fp16
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
