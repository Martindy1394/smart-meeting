#!/usr/bin/env bash
# Generate frontend TypeScript types from the FastAPI OpenAPI schema.
# Fails CI when backend field renames drift from committed types.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/frontend/src/types/openapi.d.ts"
SPEC="$ROOT/frontend/openapi.json"

cd "$ROOT/backend"
python3 - <<'PY'
from app.main import app
import json
from pathlib import Path
spec = app.openapi()
Path("../frontend/openapi.json").write_text(json.dumps(spec, indent=2))
print("wrote frontend/openapi.json")
PY

cd "$ROOT/frontend"
if ! command -v npx >/dev/null; then
  echo "npx required" >&2
  exit 1
fi
npx --yes openapi-typescript@7.4.4 "$SPEC" -o "$OUT"
echo "Generated $OUT"

# Soft check: committed models.ts should mention key fields that must stay aligned.
node - <<'NODE'
const fs = require("fs");
const models = fs.readFileSync("src/types/models.ts", "utf8");
const required = [
  "start_time",
  "end_time",
  "low_confidence",
  "extractive_fallback",
  "faithfulness",
  "custom_vocab",
  "action_items",
];
for (const key of required) {
  if (!models.includes(key)) {
    console.error(`models.ts missing required field marker: ${key}`);
    process.exit(1);
  }
}
console.log("models.ts field markers OK");
NODE
