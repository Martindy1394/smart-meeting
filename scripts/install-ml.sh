#!/usr/bin/env bash
# Install Whisper / BART / mBART Python deps for the Smart Meeting backend.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/backend"
python3 -m pip install -r requirements-ml.txt
python3 - <<'PY'
import faster_whisper, transformers, torch
print("OK: faster-whisper", faster_whisper.__version__)
print("OK: transformers", transformers.__version__)
print("OK: torch", torch.__version__)
PY
