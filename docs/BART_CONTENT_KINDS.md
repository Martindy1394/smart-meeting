# BART content kinds (`meeting` vs `general`)

## Gap A prerequisite

Lyric/song content-loss dedupe harden must be present before topic context
can be trusted:

- `_DEDUPE_MIN_SHARED_WORDS >= 3`
- `_DEDUPE_LOCAL_WINDOW >= 2`

Confirmed in `backend/tests/test_general_bart_summary.py::test_gap_a_dedupe_fix_landed`
(see also `docs/LYRIC_MT_CONTENT_LOSS.md`).

## Behavior

| `source_kind` | BART frame prefix | Output shape |
|---|---|---|
| `meeting` (default; `english_translation` alias) | `Board meeting discussion and decisions.` | Discussion / Decisions / Action items |
| `general` | `Summarize the following.` | Flat bullets or topic headings — **no** minutes regex bucketing |
| `transcript` | (raw / often PH path) | Short extractive or topic BART without minutes |

API: `POST /api/ai/summarize` body field `source_kind`. Meeting room UI sends
`"meeting"` explicitly.
