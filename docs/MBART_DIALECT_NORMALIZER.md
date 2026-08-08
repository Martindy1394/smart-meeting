# mBART dialect normalizer (SmartScribe system prompt)

## Why this is not a chat “system prompt”

`facebook/mbart-large-50-many-to-many-mmt` is a **seq2seq translation** model.
It has no chat interface and **cannot consume** an instruction block the way
GPT-style models do. Pasting `MBART_SYSTEM_PROMPT` into the encoder would
pollute the source sentence and hurt translation.

Instead, Smart Meeting stores the SmartScribe prompt and **implements its rules
as deterministic pre-mBART cleanup** in `backend/app/services/mbart_dialect.py`.

## Integration

```
Whisper ASR (hil / tl / Taglish)
  → (optional) glossary.protect
  → mbart_dialect.normalize_for_mbart   ← MBART_SYSTEM_PROMPT rules
  → mBART encode (tl_XX / proxy) → English
```

Meeting context (`source_lang`, title, participants) is bound in `/api/ai/translate`
and `/api/ai/summarize` via `mbart_dialect.meeting_context(...)` so the rendered
prompt stays available for logs/docs while cleanup uses the same metadata.

## Rules applied

| Prompt rule | Implementation |
|---|---|
| Remove fillers (`uh`, `um`, `'di ba`, `kuan`, …) | regex strip |
| Remove false starts | repeated-token collapse |
| Preserve names / numbers | glossary placeholders + number stash |
| Code-switch → formal Filipino | light Taglish verb map (`i-move`→`ilipat`, …) |
| Decisions / actions | `NAPAGPASYAHAN:` / `AKSYON:` labels when cues present |
| Do not invent facts | no LLM rewrite — cleanup only |

## Config

```bash
# backend/.env
MBART_DIALECT_NORMALIZE=true   # default
```

Disable only for ablation/debug. Health exposes `pipeline.mbart_nllb.mbart_dialect_normalize`.

## Prompt template

See `MBART_SYSTEM_PROMPT` in `mbart_dialect.py` (same text as the SmartScribe
dialect-normalizer prompt). Render with `render_system_prompt(...)`.
