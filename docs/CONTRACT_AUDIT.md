# Frontend ↔ API ↔ Database contract audit

Review date: 2026-07-30  
Layers: React client (`frontend/src`) · FastAPI schemas/routers/WS · SQLAlchemy models.

---

## How the layers connect

```
UI components
  → api/client.js  (JSON + Bearer / refresh)
  → FastAPI routers + Pydantic schemas
  → SQLAlchemy ORM  (D1 Users, D2 Meetings, D3 Segments)
  → Redis / disk audio (D4 / D5; outside relational DB)
```

| Concern | Frontend | API schema | Database |
|---|---|---|---|
| User identity | `user.id`, tokens in `localStorage` | `UserResponse`, JWT `sub` | `users.id` (String 36) |
| Meeting body | ISO dates, `attendees: string[]` | `MeetingDetail.attendees: list[str]` | `meetings.attendees` **JSON Text** |
| Transcript | `final_transcript` string | same | `Text` |
| Segments | mostly unused in UI | `start_time` / `end_time` | same columns; WS finalize uses `start`/`end` |
| AI outputs | `summary`, `translation`, `engine`, `extractive_fallback`, `faithfulness` | `SummarizeResponse` | summary/translation columns; flags **not persisted** |
| Audio | blob URL from `/audio` | file/bytes | `audio_path` + D5 files (not a BLOB column) |

---

## Critical

### C1 — WebSocket auth does not refresh the access JWT
- **Mismatch:** REST client refreshes on 401; `useRecorder` builds WS URL with `getToken()` only.
- **Impact:** Access TTL ≈ 30 minutes; multi-hour meetings lose the socket after expiry → reconnect storm / dead captions.
- **Evidence:** `frontend/src/hooks/useRecorder.js` (`buildWsUrl`); `backend/app/ws/transcription.py` close `4401`.
- **Fix:** Refresh access before `openSocket`; stop reconnect-loop on persistent 4401.

### C2 — Finalize can hang if WS dies after `stop` was sent
- **Mismatch:** REST `/meetings/{id}/stop` is only used when stop was *not* sent over WS.
- **Impact:** UI stuck on “Finalizing…”; meeting left `recording`/`processing` until janitor.
- **Evidence:** `useRecorder.stop` / `stoppingRef`.
- **Fix:** Timeout → call `stopMeetingRecording`; treat WS close during finalize as REST finalize.

### C3 — Decrypt failures look like “no audio”
- **Mismatch:** Redis/crypto errors return `b""`; FE maps 404 audio to empty state.
- **Impact:** After key rotation, playback/retranscribe fail silently while `has_audio` may still be true.
- **Evidence:** `redis_store.get_wav_bytes` / `get_pcm`; `MeetingRoom.loadAudio`.
- **Fix:** Return 500 + clear detail; FE show decrypt/config error, not “No recording yet”.

---

## Medium — types, validation, exceptions

### M1 — Segment timestamp field names disagree
- Finalize/WS payload: `start` / `end`
- ORM + REST: `start_time` / `end_time`
- Live persist often leaves times at `0.0`
- **Impact:** Contract confusion; weak timestamped export for live-only data.

### M2 — Meeting details required in UI, optional in API/DB
- FE blocks Start without title/venue/datetime/≥1 attendee.
- `MeetingCreate` allows `title=""`, `venue=""`, `attendees=[]`.
- ORM default `"Untitled meeting"` bypassed when create writes `""`.
- **Impact:** API clients can create incomplete meetings.

### M3 — Attendees: `list[str]` in API, JSON `Text` in DB
- Handled by `_clean_attendees` / `_parse_attendees`, but raw ORM is a string; search uses `ilike` on JSON text.
- **Impact:** Fragile if any path skips the helper; odd search matches on quotes/brackets.

### M4 — Upload ASR language hard-coded `"auto"`
- Retranscribe uses `effective_asr_language(meeting.language)`; upload background task always `"auto"`.
- **Impact:** Inconsistent language path for API upload clients.

### M5 — WS close codes / lock errors under-handled on FE
- Server: `4401` auth, `4409` lock.
- FE reconnect logic does not special-case them → possible reconnect loops on second tab.

### M6 — `pagehide` stop may send expired Bearer
- Keepalive `/stop` uses access token only (no refresh) → orphan `recording` until janitor.

### M7 — FE swallows errors as empty state
- `loadMeetings` → `[]` on any failure; audio errors look like no file; create/delete often silent.
- **Impact:** 401/500 indistinguishable from “no data”.

### M8 — Profile update validation weaker than signup
- Signup requires non-empty names/position/workplace; profile PATCH allows clearing them to `""`.

### M9 — Language validation asymmetric
- Create accepts any ≤16-char string; update allowlists; product always forces `auto` after ASR.

### M10 — Default export `pdf` + Latin-1 sanitizer
- PH characters degraded in PDF while FE defaults to PDF.

### M11 — Summarize race after retranscribe
- Persist clears summary/translation; UI effect + explicit summarize can double-fire.

### M12 — Blob helpers (export/audio) 401 retry weaker than `request()`
- Detail array flattening inconsistent; retry guard less strict.

### M13 — Auth bootstrap clears tokens on any `/me` failure
- Network blip can log the user out even with a valid refresh token.

---

## Low / intentional drift

| Item | Notes |
|---|---|
| `engine` vs internal `summary_engine` | Wire field is combined `engine` — FE aligned |
| `expires_in` unused on FE | No proactive refresh scheduling |
| Logout `204` | Handled correctly |
| Hidden ORM fields | `hashed_password`, `audio_path`, `is_active` not exposed — OK |
| `has_audio` / `has_transcript` | Computed in router, not DB columns — OK if documented |
| `summary_format=""` in DB | FE defaults to `"bullets"` |
| No FE for upload / `has_audio` list filter | API-only features |
| Live segment DB write errors swallowed | Captions still stream; history may be sparse |

---

## Validation matrix (selected fields)

| Field | FE | API | DB | Verdict |
|---|---|---|---|---|
| Password (signup) | Regex-ish UX | `_PASSWORD_RE` | hash | Aligned |
| Username | Client checks | Regex + unique | unique index | Aligned |
| Meeting title | Required before Start | Optional `""` | default unused | **Asymmetric** |
| Attendees | `string[]`, ≥1 before Start | `list[str]`, may be `[]` | JSON text | Shape OK; rules asymmetric |
| `meeting_date` | `toISOString()` | `datetime` | tz DateTime | Aligned |
| `output_format` | bullets/numbered toggle | enum validator | `summary_format` string | Aligned |
| Export format | pdf/docx/txt select | regex query | n/a | Aligned; PDF charset weak |
| `extractive_fallback` / `faithfulness` | UI state | response fields | **not stored** | OK for session; lost on reload |
| Segment times | unused in UI | `start_time`/`end_time` | same | WS finalize uses different names |

---

## Unhandled exception hotspots

1. Redis decrypt → empty bytes → FE “no audio” (C3).
2. Background retranscribe → `failed` only visible if UI is polling.
3. Live `_persist_live_segment` rollback with no client signal.
4. Workspace list/create/delete catch without user-visible error (M7).
5. WS finalize without REST fallback timeout (C2).

---

## Recommended fix order

1. WS token refresh + 4401/4409 handling (C1, M5).
2. Finalize watchdog → REST `/stop` (C2, M6).
3. Decrypt/audio error surfacing (C3, M7).
4. Align `MeetingCreate` validation with FE details gate (M2).
5. Unify segment timestamp names; persist live times (M1).
6. Upload language = meeting language (M4).
7. PDF Unicode or default export to DOCX (M10).
8. Persist or re-fetch `extractive_fallback` / faithfulness if needed after reload.

Related: [`TEST_PLAN.md`](TEST_PLAN.md) gaps E1–E17, [`DFD.md`](DFD.md).
