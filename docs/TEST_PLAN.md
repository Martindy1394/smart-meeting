# Smart Meeting — End-to-End Trace, Edge Cases, and Test Plans

Canonical stores (see [`DFD.md`](DFD.md)): **D1** Users · **D2** Meetings ·
**D3** Segments · **D4** Redis · **D5** Disk audio.

---

## 1. End-to-end traces

### Journey A — Auth session

```
UI Login/Signup
  → POST /api/auth/login|signup
  → D1 Users (verify/create)
  → issue access (≈30m) + refresh (≈7d)
  → localStorage sm_token / sm_refresh
  → GET /api/auth/me
  → [later 401] POST /api/auth/refresh (rotate; revoke old jti → D4 denylist)
  → POST /api/auth/logout (revoke jtis)
```

**Key files:** `routers/auth.py`, `security.py`, `frontend/src/api/client.js`,
`AuthContext.jsx`

### Journey B — Live meeting → refined transcript → English → minutes

```
POST /api/meetings (D2 status=recording)
  → PATCH details (title, venue, datetime, attendees)  [UI gate before Start]
  → WS /ws/transcribe?token&meeting_id
      → auth + ownership + live lock (D4)
      → PCM → D5 .pcm + D4 rolling buffer
      → live Whisper windows → captions + D3 live segments
  → [optional client pause: mute track, drop PCM; server does not freeze ASR drain]
  → stop (WS {type:stop} or POST /meetings/{id}/stop)
      → finalize: D5 WAV (optional encrypt) → Whisper final
      → D2 final_transcript + D3 final segments; clear live PCM
  → POST /api/ai/summarize (force_retranslate after finalize)
      → NLLB/mBART → D2 translation
      → BART minutes → D2 summary (+ faithfulness / extractive flags)
  → UI: transcript, English, minutes, playback
```

**Key files:** `ws/transcription.py`, `finalize.py`, `services/transcription.py`,
`routers/ai.py`, `MeetingRoom.jsx`, `useRecorder.js`

### Journey C — Upload / retranscribe (API-first)

```
POST /api/meetings/{id}/audio?transcribe=true|false
  → save WAV (D5; Redis WAV if small)
  → optional background Whisper (language forced "auto" today)
POST /api/meetings/{id}/retranscribe
  → ensure WAV → processing → Whisper → persist (clears prior summary/translation)
  → UI poll → auto summarize/translate
```

**Note:** There is **no upload control in the React UI**; upload is API-only.

### Journey D — History, search, export, playback

```
GET /api/meetings?search=…  (ILIKE title/venue/attendees/transcript/summary/translation)
  → GET /api/meetings/{id}
  → GET /api/meetings/{id}/export?format=txt|docx|pdf
  → GET /api/meetings/{id}/audio  (decrypt if SMENC1; full body Response)
```

### Journey E — Encryption-at-rest

```
DATA_ENCRYPTION_KEY set
  → finalize/upload WAV write: SMENC1 + Fernet (D5, D4 WAV)
  → Redis rolling PCM: decrypt–append–encrypt
  → live D5 .pcm remains plaintext until deleted after finalize
  → reads decrypt transparently for ASR/playback
```

### Journey F — Degraded / failure paths

| Condition | Observed behavior |
|---|---|
| Whisper missing | Live: audio saved, no captions; finalize may use live-caption fallback or `failed`; upload prepare **503** |
| LLM/transformers missing | Summarize → extractive + UI banner; Translate **503** |
| Redis down | Disk PCM continues; no shared live lock / shared JWT denylist; reconnect limited |
| Empty recording | Finalize → empty or live-only transcript (`empty-recording`) |
| Silence | Live ASR skipped by energy gate |
| Bad/foreign JWT | API **401**; non-owner meeting **404**; WS close **4401** |

---

## 2. Missing / weak edge cases (gaps)

Prioritized for test design and future fixes:

| ID | Gap | Risk | Evidence |
|---|---|---|---|
| E1 | Multi-tab live write when Redis down (lock always allows) | Corrupted/interleaved PCM | `redis_store.acquire_live_lock` |
| E2 | JWT denylist process-local without Redis | Revoked token still works on another worker | `security._local_denylist` |
| E3 | WS token in query string | Token leakage via logs/proxies | `useRecorder.js`, `ws/transcription.py` |
| E4 | Upload background ASR always `"auto"` vs meeting language on retranscribe | Inconsistent language path | `meetings.upload_meeting_audio` |
| E5 | Pause is client-only; no pause markers; ASR may drain backlog | Confusing timeline | `useRecorder.js`, WS `pause` handler |
| E6 | `pagehide` keepalive stop can fail if access JWT expired | Mitigated by live proactive refresh; residual if refresh down at unload | `useRecorder.js` |
| E7 | Full WAV loaded into memory for playback | OOM / slow on multi-hour files | `get_meeting_audio` |
| E8 | PDF export Latin-1 lossy for PH text | Bad PDF for Hiligaynon/Tagalog | `export._pdf_safe` |
| E9 | No UI for audio upload | Feature discoverability / E2E gap | Frontend vs `POST .../audio` |
| E10 | Auto-summarize race after retranscribe clears summary | Flicker / stale summarize | `persist_transcript` + `MeetingRoom` effects |
| E11 | Translate has no extractive fallback | Hard fail without ML | `ai.translate` |
| E12 | Live segment timestamps often 0 | Weak timestamped export for live-only | `_persist_live_segment` |
| E13 | Create meeting allows empty title via API | Incomplete meetings | `MeetingCreate` |
| E14 | Stuck `processing` until stale lease (~45m) | Poor UX if worker dies | `janitor` / `is_processing_stale` |
| E15 | Concurrent stop (WS + REST + janitor) | `lease-lost` / empty client expectation | `finalize` claim |
| E16 | In-memory rate limits not shared across workers | Weak auth/AI throttling in multi-process | `limiter.py` |
| E17 | Encrypted Redis read errors can look like “no audio” | Ops misdiagnosis after key rotation | `redis_store.get_wav_bytes` |

---

## 3. Test plans

### 3.1 Test strategy

| Layer | Goal | Tools |
|---|---|---|
| Unit | Pure functions, crypto, faithfulness, language forcing, export renderers | `backend/tests/*.py` |
| API integration | Auth, meetings CRUD, summarize/translate, export, health | FastAPI `TestClient` |
| WS / live | Capture, captions, stop, unauthorized, lock | `websockets` / pytest-asyncio (to add) |
| UI / E2E | Full Journey B in browser | Playwright/Cypress (to add) or manual script below |
| Eval / offline | ASR WER, MT tags, Hiligaynon human sheet | `wer.py`, `benchmark_mbart_tags.py` |
| Ops / security | Encryption key, TLS, multi-worker revoke | Staging checklist |

Existing automated coverage (baseline):  
`test_review_hardening`, `test_health_pipeline`, `test_meeting_export`,
`test_transcription_reliability`, `test_topic_summarize`,
`test_session_audio_storage`, `test_backend_reliability`, PLD/RNNT/WER tests.

### 3.2 Auth test plan

| Case | Steps | Expected |
|---|---|---|
| A1 Signup happy | Valid payload | 201 + access + refresh + user in D1 |
| A2 Weak password | Missing digit/special | 422 |
| A3 Duplicate username/email | Second signup | 409 |
| A4 Login bad password | Wrong password | 401 vague message |
| A5 Refresh rotate | Use refresh once, reuse old | First OK; old refresh 401 |
| A6 Logout revoke | Logout then call `/me` with old access | 401 |
| A7 Refresh after logout | Logout with refresh body, then refresh | 401 |
| A8 Multi-worker revoke* | Logout on worker A, `/me` on worker B **without Redis** | **Known fail (E2)** — document; with Redis should 401 |
| A9 Rate limit | Burst login | 429 after limit |

### 3.3 Live meeting E2E test plan (manual + future Playwright)

| Case | Steps | Expected |
|---|---|---|
| B1 Happy path | Create → fill details → record 30–60s speech → stop | Live captions; status finalized; transcript non-empty; English + minutes appear |
| B2 Details gate | Start with empty title | UI blocks Start |
| B3 Hiligaynon never `tl` | Record Ilonggo; inspect health/logs/decode | No forced `tl`; auto-detect + prompt path |
| B4 Pause/resume | Pause 10s silence, resume speak | No long silence caption spam; timeline continuous (note E5) |
| B5 Two tabs same meeting | Start in tab A and B with Redis up | Second gets lock conflict (4409) |
| B6 Two tabs Redis down* | Same without Redis | **Known risk (E1)** — interleaved PCM |
| B7 Stop via REST | Kill WS mid-recording; `POST /stop` | Finalize still runs |
| B7b Stop after WS drop | Record → Stop → kill WS before `final_transcript` | FE arms watchdog / onclose → REST `/stop`; leaves Finalizing…; meeting finalized |
| B7c Long-meeting JWT | Record > access TTL (~30m) or force near-expiry token | Proactive refresh; reconnect uses fresh JWT; no 4401 reconnect storm |
| B7d Auth close 4401 | Connect WS with expired access (valid refresh) | FE refreshes, reconnects ≤2; captions resume |
| B7e Lock close 4409 | Second tab starts same meeting | Error message; no reconnect loop |
| B8 Max duration soft-cap | Approach `MAX_MEETING_HOURS` | Warning / stop behavior per config |
| B9 Silence-only | Record mute room | Little/no junk transcript; not catastrophic loops |
| B10 Code-switch | Hiligaynon + English sentences | Transcript retains both; EN translation usable |
| B11 Extractive fallback | Stop API with ML uninstall / BART fail | Banner “Extractive fallback”; not labeled as full BART |
| B12 Faithfulness warn | Force summary with hallucinated action (unit/API) | `faithfulness.status=warn` + UI list |

### 3.4 Upload / retranscribe test plan

| Case | Steps | Expected |
|---|---|---|
| C1 Upload WAV + transcribe | `POST .../audio?transcribe=true` | `processing` → finalized transcript |
| C2 Upload empty | 0-byte body | 400 |
| C3 Upload oversized | >60 MB | 413 |
| C4 Upload without Whisper | ML missing | 503 on prepare |
| C5 Retranscribe clears AI | Meeting with summary; retranscribe | summary/translation cleared then regenerated by UI |
| C6 Language consistency* | Meeting language `tl`; upload transcribe | **Gap E4** — today upload uses `"auto"`; assert desired policy once fixed |
| C7 UI upload | Look for upload control | **Missing (E9)** — track as product test fail until UI added |

### 3.5 Search / export / playback test plan

| Case | Steps | Expected |
|---|---|---|
| D1 Search transcript keyword | Search unique word in transcript | Meeting listed |
| D2 Search other owner | User B searches User A content | Not visible |
| D3 Export TXT/DOCX/PDF | All three formats | Non-empty; DOCX zip/`PK`; PDF `%PDF` |
| D4 Export PH characters | Hiligaynon in transcript | TXT/DOCX preserve; PDF may degrade (**E8**) |
| D5 Export partial | Transcript only, no summary | Placeholders, still downloads |
| D6 Playback encrypted | Key set; play meeting | Audio plays after decrypt |
| D7 Playback wrong/missing key | Ciphertext WAV, key removed | Clear error, not silent empty (**watch E17**) |
| D8 Long audio playback* | Multi-hour WAV | **Risk E7** — latency/memory; prefer range/stream later |

### 3.6 Encryption / privacy test plan

| Case | Steps | Expected |
|---|---|---|
| F1 Round-trip | Encrypt/decrypt WAV bytes | Plaintext restored; file starts with `SMENC1` on disk |
| F2 Legacy plaintext | Old WAV without header | Still readable |
| F3 Redis PCM encrypt mode | Append chunks with key | get_pcm returns concatenated plaintext |
| F4 Health flag | `/api/health` | `encryption_at_rest.enabled` true/false |
| F5 Live PCM plaintext | During recording inspect `.pcm` | Plaintext (documented limitation) |
| F6 DB text not encrypted | Inspect SQLite transcript | Readable (documented; TDE/volume for ops) |

### 3.7 AI / model test plan

| Case | Steps | Expected |
|---|---|---|
| G1 Summarize without transcript | No finalize | 400 |
| G2 Translate without ML | Call translate | 503 |
| G3 MT tag benchmark protocol | `benchmark_mbart_tags.py --fixtures-only` | `report.json` + Hiligaynon worksheet |
| G4 MT tag live (GPU) | Full benchmark | `tl_XX` vs `id_ID` scores; no invented Hiligaynon scores |
| G5 WER eval | Labeled clip + `wer.py` | Numeric WER recorded (team metric) |
| G6 Language forcing unit | `_forced_language("hil"|"auto")` | `None` |

### 3.8 Non-functional / ops checklist

| Case | Check |
|---|---|
| N1 | Production refuses default `JWT_SECRET_KEY` |
| N2 | TLS terminated in front of API |
| N3 | Redis required in multi-worker prod (locks + denylist) |
| N4 | `DATA_ENCRYPTION_KEY` in secrets manager; backup/rotation runbook |
| N5 | Disk space for 8h+ PCM/WAV |
| N6 | `/api/health` pipeline + auth + encryption fields monitored |
| N7 | Janitor clears abandoned `recording` / stale `processing` |

---

## 4. Suggested automation backlog (ordered)

1. API tests for auth refresh/logout revoke (extend `test_review_hardening`).
2. API tests for meetings search ownership + export PH text (TXT vs PDF).
3. Finalize empty/silence/live-fallback cases (extend `test_backend_reliability`).
4. WS integration: auth failure 4401, second-tab 4409 with Redis.
5. Playwright Journey B smoke (create → record fixture audio → stop → summary).
6. Fix-then-test: E4 language on upload, E7 streamed audio, E9 upload UI, E8 PDF Unicode font.

---

## 5. Traceability to requirements (DFD)

| DFD process | Covered by journeys | Primary tests |
|---|---|---|
| 1.0 Authenticate | A | §3.2 |
| 2.0 Capture | B | §3.3 B1–B9 |
| 3.0 Transcribe live | B | §3.3, G6 |
| 4.0 Transcribe final | B, C | §3.3–3.4 |
| 5.0 Translate | B, F | §3.3 B11, §3.7 G2 |
| 6.0 Summarize | B | §3.3 B11–B12, topic tests |
| 7.0 Search & manage | D | §3.5 D1–D2 |
| 8.0 Export | D | §3.5 D3–D5 |

This document is the master test plan for Smart Meeting’s current shipped scope;
items marked with `*` are known gaps to track as defects or explicit limitations
rather than silent assumptions.
