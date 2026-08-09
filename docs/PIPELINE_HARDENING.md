# Pipeline hardening (Tier 1–4)

Hardening pass over the existing Hiligaynon-biased, English-minutes design.
Diarization and client E2E encryption remain out of scope.

## Tier 1 — ASR quality

| Item | Implementation |
|---|---|
| VAD gate | `services/vad.py` — webrtcvad when installed, else energy+ZCR. Called before live and final Whisper. |
| Session language lock | WS caches detection after ~8s of speech in Redis + `Meeting.language_locked`. Never forces Hiligaynon→`tl`. |
| Confidence filtering | `avg_logprob` / `no_speech_prob` / `low_confidence` on `TranscriptSegment`. Hard-drop high `no_speech_prob`; flag low logprob. |
| Custom vocabulary | `Meeting.custom_vocab` (+ attendees) appended to Whisper `initial_prompt`. |

## Tier 2 — Translation & minutes

| Item | Implementation |
|---|---|
| Transcript↔translation faithfulness | `llm.assess_translation_faithfulness` (+ glossary term survival). Persisted as `translation_faithfulness_json`. |
| Do-not-translate glossary | `services/glossary.py` protect/restore placeholders around NLLB/mBART. |
| Structured action items | `services/action_items.py` extracts `{owner, action, due_date}` from BART Action Items. |

## Tier 3 — Backend architecture

| Item | Implementation |
|---|---|
| Background jobs | `services/jobs.py` + `/api/jobs/*` (finalize / retranscribe, dedupe keys). Inline Redis worker by default. |
| OpenAPI→TS | `scripts/generate-frontend-types.sh` + `npm run typecheck:models`. |
| Stage metrics | `services/pipeline_metrics.py`; exposed on `/api/health`. |
| Audio retention | `AUDIO_RETENTION_DAYS` (default 30); janitor purges WAV/PCM, keeps text. |

## Tier 4 — Frontend

| Item | Implementation |
|---|---|
| IndexedDB PCM buffer | `useRecorder.js` persists PCM while WS is down; flushes on reconnect. |
| Live confidence UI | Wavy underline + badge when `liveLowConfidence`; final segments mark `low_confidence`. |

## Tests

`backend/tests/test_pipeline_hardening.py` — silence VAD, confidence drop/flag, glossary spelling, translation faithfulness, action-item extraction, custom vocab prompt.
