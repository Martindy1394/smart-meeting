#!/usr/bin/env bash
# Convert a Hugging Face Whisper fine-tune to CTranslate2 for faster-whisper live ASR.
#
# Usage:
#   ./scripts/hiligaynon_asr/export_ct2.sh ./models/whisper-medium-hiligaynon ./models/whisper-medium-hiligaynon-ct2
#
# Then set:
#   WHISPER_LIVE_HILIGAYNON_MODEL=/abs/path/to/whisper-medium-hiligaynon-ct2
set -euo pipefail

SRC="${1:-}"
DST="${2:-}"
if [[ -z "$SRC" || -z "$DST" ]]; then
  echo "Usage: $0 <hf-whisper-dir-or-repo> <ct2-output-dir>" >&2
  exit 1
fi

python3 -m pip install -q "ctranslate2>=4.0" "transformers>=4.40"
ct2-transformers-converter \
  --model "$SRC" \
  --output_dir "$DST" \
  --copy_files tokenizer_config.json preprocessor_config.json \
  --quantization int8

echo "Exported CT2 model to $DST"
echo "Set WHISPER_LIVE_HILIGAYNON_MODEL=$(cd "$DST" && pwd)"
