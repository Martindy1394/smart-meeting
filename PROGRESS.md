# Session progress — Smart Meeting (minute-making app)

This note records what was discussed and implemented in the Cursor cloud-agent
session that produced the stacked branches ending at
`cursor/gpu-faster-whisper-78af` (11–12 Sep 2026). It is a working log, not a
product spec. Canonical product claims live in [`docs/PRODUCT.md`](docs/PRODUCT.md)
and [`README.md`](README.md).

**Repo:** [Martindy1394/smart-meeting](https://github.com/Martindy1394/smart-meeting)  
**Local clone used by the operator:** `M:\MSCS\MY THESIS\Technical`  
**UI:** `http://127.0.0.1:5173` · **API:** `http://127.0.0.1:8000`

---

## 1. What the product is

Smart Meeting (also referred to as SmartScribe in discussion) is a FastAPI +
React app that:

1. Captures live 16 kHz mono PCM in the browser (`AudioWorklet`).
2. Streams it over WebSocket to the API for **live Whisper** captions.
3. On stop, runs a **full-accuracy Whisper** pass on the saved WAV.
4. Translates to English (**NLLB** for Tagalog; **Google Translate** for
   Hiligaynon when configured, else NLLB fallback; **mBART** as a broader MT
   path).
5. Summarizes English minutes with **BART** (bullets or numbered:
   Discussion / Decisions / Action items).
6. Stores per-user meetings in SQLite by default (Postgres supported).

Meeting language in the UI is always **auto**, biased toward Hiligaynon
prompts and Philippine models. There is no Spoken-language picker.

---

## 2. Operating constraints discussed in this session

### Cloud agent vs the operator’s GPU

The cloud agent **cannot** use the NVIDIA GPU on the Windows thesis machine.
Whisper/CTranslate2 only see a GPU inside the **local uvicorn process**. The
browser on port 5173 never runs Whisper.

```
Mic → Vite UI (5173) → FastAPI on the Windows PC (8000)
                         → faster-whisper / CTranslate2
                         → GPU on that same PC (if CUDA is enabled there)
```

A remote agent can change code and settings; CUDA only takes effect after the
operator restarts the API on the machine that has the driver.

### Windows Python / models

Discussed (and still relevant for local setup):

- `faster-whisper` must be installed in the **same Python** that runs uvicorn.
  A “Whisper unavailable” banner usually means that process lacks the package
  or was not restarted after install.
- `torch==2.5.1` has no Windows Python 3.13 wheel; the operator had a CUDA
  torch (e.g. `2.7.1+cu118`).
- `sentencepiece` 0.2.0 can fail to build on Windows 3.13. Whisper itself does
  not need sentencepiece; mBART does.
- The cloud VM used for this session had **no NVIDIA GPU**. Health checks there
  reported Whisper/BART/NLLB available on CPU.

### How to run locally (two processes)

From the Technical clone:

```bat
cd "M:\MSCS\MY THESIS\Technical\backend"
py -3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir app
```

```bat
cd "M:\MSCS\MY THESIS\Technical\frontend"
npm run dev
```

Repo-root launchers added this session: `start-api.bat`, `start-frontend.bat`,
`start-all.bat`. Copy `backend\.env.example` → `backend\.env`. Redis at
`localhost:6379` is optional (in-process fallback if Redis is down).

A Cursor “Unable to reconnect workspace” message discussed in session was a
**desktop-to-cloud-agent session drop**, not an application crash. The agent
could still be running.

---

## 3. Pipeline as it stands after this session

| Stage | Behavior |
|---|---|
| Live ASR | `faster-whisper` (default `small`), ~10 s windows / 5 s hop; optional Tagalog RNN-T if NeMo is installed |
| Final ASR | Stronger pass (`medium` and/or HF PH models such as `rbcurzon/whisper-medium-ph`, Tagalog HF) |
| Device | `WHISPER_DEVICE=auto` → CUDA if the **API host** has a GPU, else CPU |
| Compute | `WHISPER_COMPUTE_TYPE=auto` → **float16** on CUDA, **int8** on CPU. Explicit `int8` on GPU → CTranslate2 **int8_float16** |
| Speakers | Spectral clustering of PCM, then **Voice 1–3 ranked by Whisper accuracy** (Voice 1 = most accurate). Not named enrollment |
| Transcript UI / export | Lead with `Voice N:` chips/lines; `[start–end]` times remain on stored segments |
| Meeting form | Title*, Venue*, Presiding officer (optional), Date/time*, Attendees* |
| Name suggestions | `GET /api/meetings/suggestions` — officers and attendees from prior meetings (newest first, casefold unique) |
| Search | Title, venue, attendees, transcript, summary, translation, presiding officer |
| Minutes | BART bullets/numbered; action items extracted when present |
| Translation | EN passthrough; Tagalog → NLLB; Hiligaynon → Google if configured else NLLB |

Prompt hints for Whisper now come from **attendee names + presiding officer**,
not a free-text custom vocabulary field.

---

## 4. Work implemented (by theme)

Work landed as stacked git branches / PRs off `main` through
`cursor/build-minute-making-app-78af` and later feature branches. Dates below
are commit dates on those branches (11 Sep 2026 unless noted).

### 4.1 Windows launchers

- Added `start-api.bat`, `start-frontend.bat`, `start-all.bat` at the repo root
  so the operator does not have to remember PowerShell `cd` paths.
- API script prefers a venv if present; otherwise `py -3 -m uvicorn ...`.
- Frontend script can `npm install` once if `node_modules` is missing.

**PR:** [#6](https://github.com/Martindy1394/smart-meeting/pull/6)

### 4.2 Transcript UI vs README (search / timestamps / export)

- Aligned the meeting room and history with the README: searchable keywords,
  export, and transcript surfaces.
- History search includes meeting metadata and body text (later also
  presiding officer).

**PR:** [#7](https://github.com/Martindy1394/smart-meeting/pull/7)

### 4.3 Anonymous voice labels (instead of clock-led UI)

**Problem discussed:** clock stamps (`[00:01–00:04]`) made the transcript look
like a log, not minutes. The product already had lightweight diarization.

**Implemented:**

- `backend/app/services/live_speakers.py` — spectral fingerprint (pitch +
  log-mel bands + RMS), up to `LIVE_MAX_VOICES` (default 3).
- Live WS windows and final WAV slices get `speaker_index` / `speaker_label`.
- UI chips: Voice 1 / Voice 2 / Voice 3. Consecutive same-voice turns merge.
- Exports use `Voice N: text` under a voice-labeled transcript section.
- Labels are **anonymous**. They are not attendee names. Clustering can split
  one person into two voices.

**PRs:** [#5](https://github.com/Martindy1394/smart-meeting/pull/5) (live labels),
[#8](https://github.com/Martindy1394/smart-meeting/pull/8) (voice over timestamps)

### 4.4 Voice labels ranked by ASR accuracy

**Request:** determine / set / base voice labels on **voice accuracy**.

**Implemented:**

- Clustering still groups the same talker (stable internal cluster id).
- Each cluster is scored from Whisper `avg_logprob`, `no_speech_prob`, and the
  low-confidence flag.
- Display **Voice 1** = highest mean accuracy, Voice 2 next, and so on.
- Live: `bind_asr_accuracy` after each window so ranking can update as more
  speech arrives.
- Final pass: `label_segments` ranks all segments together.
- If no confidence scores yet, labels keep **first-seen** order.
- UI chip `title` explains the ranking (highest / second-highest / lower).

**PR:** [#15](https://github.com/Martindy1394/smart-meeting/pull/15)

### 4.5 Meeting details: office → officer → suggestions

Sequence of product-form changes:

1. **Presiding office** field added, then replaced by **Presiding officer**,
   then **office removed** from the form (officer kept).
2. Lightweight SQLite migrations in `backend/app/database.py` add columns
   (`presiding_office` may still exist unused on old DBs; `presiding_officer`
   is used).
3. **Directory suggestions:** `collect_name_directory` in
   `backend/app/services/attendees.py`; route `GET /api/meetings/suggestions`
   declared **before** `GET /{meeting_id}`. UI: chips + `<datalist>` on officer
   and attendee inputs. Suggestions only appear after saved meetings contain
   names.

**PRs:** [#9](https://github.com/Martindy1394/smart-meeting/pull/9),
[#10](https://github.com/Martindy1394/smart-meeting/pull/10),
[#11](https://github.com/Martindy1394/smart-meeting/pull/11),
[#12](https://github.com/Martindy1394/smart-meeting/pull/12)

Required fields remain: Title, Venue, Date & time, at least one attendee.
Autosave (~750 ms) when complete and dirty. Spoken language is always sent as
`auto`.

### 4.6 Custom vocabulary removed (UI then system)

**Request:** screenshot of “Custom vocabulary (optional)” → remove it; then
remove it from the **code/structure**, not only the form.

**UI (PR [#13](https://github.com/Martindy1394/smart-meeting/pull/13)):**

- Textarea removed from `MeetingDetails.jsx`.
- PATCH no longer sent `custom_vocab` (so old DB values were not wiped yet).

**System (PR [#14](https://github.com/Martindy1394/smart-meeting/pull/14)):**

- Dropped from SQLAlchemy `Meeting`, Pydantic create/update/detail, frontend
  `models.ts`, OpenAPI field-marker scripts.
- SQLite lightweight migration attempts `ALTER TABLE meetings DROP COLUMN custom_vocab`.
- `parse_custom_vocab` renamed to `parse_prompt_terms`;
  `meeting_prompt_terms(meeting)` uses attendees + presiding officer.
- Live WS, finalize, and retranscribe jobs use those names as Whisper
  `initial_prompt` hints (capped, short — Whisper echoes long prompts).

`translation_glossary_json` was **kept**: MT still uses
`services/glossary.py` plus attendee names as do-not-translate terms. There is
no glossary editor in the meeting form.

### 4.7 GPU / faster-whisper / quantization (discussed then implemented)

**Inquiry (no code at first):** how can the agent use the operator’s GPU for:

1. Faster-Whisper / CTranslate2 (~4× vs stock OpenAI Whisper)
2. CUDA + FP16 / Tensor Cores
3. INT8 quantization

**Answer given:** the agent cannot attach to that GPU. The app already used
`faster-whisper`. Device/compute were still defaulting to `cpu` / `int8` on
the then-current branch.

**Then “update the repository” (PR [#16](https://github.com/Martindy1394/smart-meeting/pull/16)):**

- Defaults: `whisper_device = "auto"`, `whisper_compute_type = "auto"`,
  `mbart_device = "auto"`.
- `resolve_whisper_device()` — CUDA via `torch.cuda.is_available()` or
  `ctranslate2.get_cuda_device_count()`; explicit `cuda` falls back to CPU if
  none.
- `resolve_whisper_compute_type()`:
  - auto + CUDA → `float16`
  - auto + CPU → `int8`
  - `int8` + CUDA → `int8_float16` (quantized weights, FP16 compute)
- `WhisperModel(...)` load failure on CUDA retries **CPU int8**.
- Hugging Face final Whisper pipelines use CUDA + `torch.float16` when
  resolved device is CUDA.
- mBART: `resolve_mbart_device()`, `.to("cuda")`, generate tensors moved to
  the model device. BART summarizer stays on **CPU** (`device=-1`) to save
  VRAM for ASR.
- `GET /api/health` `pipeline.whisper.hardware_hint` reports
  `setting=` / `resolved=` / `compute_type=`.
- Tests: `backend/tests/test_gpu_whisper.py` (no model download).

**Operator action:** if `backend\.env` still has `WHISPER_DEVICE=cpu`, change
to `auto` (or recopy `.env.example`) and restart uvicorn on the GPU PC.

---

## 5. Pull request stack (this workstream)

| PR | Branch | Topic |
|---|---|---|
| [#4](https://github.com/Martindy1394/smart-meeting/pull/4) | `cursor/gpu-whisper-mbart-78af` | Earlier auto-CUDA (base `main`; later re-applied on the stacked tip as #16) |
| [#5](https://github.com/Martindy1394/smart-meeting/pull/5) | `cursor/live-voice-labels-78af` | Live Voice 1/2/3 |
| [#6](https://github.com/Martindy1394/smart-meeting/pull/6) | `cursor/windows-start-bats-78af` | Windows `.bat` launchers |
| [#7](https://github.com/Martindy1394/smart-meeting/pull/7) | `cursor/fix-frontend-backend-78af` | Transcript / search / export vs README |
| [#8](https://github.com/Martindy1394/smart-meeting/pull/8) | `cursor/voice-labels-over-timestamps-78af` | Voice chips instead of clock stamps |
| [#9](https://github.com/Martindy1394/smart-meeting/pull/9) | `cursor/presiding-office-field-78af` | Presiding office field |
| [#10](https://github.com/Martindy1394/smart-meeting/pull/10) | `cursor/presiding-officer-field-78af` | Presiding officer field |
| [#11](https://github.com/Martindy1394/smart-meeting/pull/11) | `cursor/remove-presiding-office-78af` | Remove office; keep officer |
| [#12](https://github.com/Martindy1394/smart-meeting/pull/12) | `cursor/directory-suggestions-78af` | Name suggestions from past meetings |
| [#13](https://github.com/Martindy1394/smart-meeting/pull/13) | `cursor/remove-custom-vocab-ui-78af` | Remove Custom vocabulary textarea |
| [#14](https://github.com/Martindy1394/smart-meeting/pull/14) | `cursor/strip-custom-vocab-78af` | Remove `custom_vocab` from API/DB/ASR |
| [#15](https://github.com/Martindy1394/smart-meeting/pull/15) | `cursor/voice-labels-by-accuracy-78af` | Rank Voice N by ASR accuracy |
| [#16](https://github.com/Martindy1394/smart-meeting/pull/16) | `cursor/gpu-faster-whisper-78af` | Auto CUDA FP16 / INT8 for faster-whisper |

Long-lived umbrella: `cursor/build-minute-making-app-78af` (historically PR #1).
Related on `main`: `cursor/mbart-ph-finetune-78af` (mBART PH fine-tune path).

To take **all** of the above locally, check out the **tip** of the stack
(`cursor/gpu-faster-whisper-78af` / PR #16), not an older middle PR.

---

## 6. Key files touched (map)

| Area | Files |
|---|---|
| Meeting form | `frontend/src/components/MeetingDetails.jsx` |
| Transcript chips | `frontend/src/components/MeetingRoom.jsx` |
| Live WS + persist | `frontend/src/hooks/useRecorder.js`, `backend/app/ws/transcription.py` |
| Voice clustering / ranking | `backend/app/services/live_speakers.py` |
| ASR load / device | `backend/app/services/transcription.py`, `backend/app/config.py` |
| Finalize / jobs | `backend/app/services/finalize.py`, `backend/app/routers/jobs.py` |
| Meetings API | `backend/app/routers/meetings.py`, `backend/app/schemas.py`, `backend/app/models.py` |
| SQLite columns | `backend/app/database.py` |
| Name directory | `backend/app/services/attendees.py` |
| mBART device | `backend/app/services/llm.py` |
| Health hints | `backend/app/main.py` |
| Types | `frontend/src/types/models.ts` |
| Env | `backend/.env.example` |
| Tests | `backend/tests/test_live_speakers.py`, `test_gpu_whisper.py`, `test_pipeline_hardening.py`, `test_model_alignment.py` |

---

## 7. Verification done in-session

- Backend unit tests on the cloud VM: **45 passing** after the GPU auto-CUDA
  change (includes live-speaker ranking and GPU resolution tests).
- Custom-vocabulary UI: meeting-details screenshot after removal (Title,
  Venue, Officer, Date/time, Attendees only).
- Earlier smoke (cloud CPU, no NVIDIA): live Whisper on a short WAV, BART
  bullets, NLLB Tagalog→EN, Hiligaynon→EN via NLLB fallback when Google was
  not configured, English passthrough.
- GPU path was **not** executed on the operator’s card from the cloud agent.
  Confirm on Windows: `GET /api/health` → `hardware_hint` should show
  `resolved=cuda` and `compute_type=float16` (or `int8_float16`) after a local
  restart with a working CUDA stack.

---

## 8. Follow-ups still on the operator / product

1. Pull `cursor/gpu-faster-whisper-78af`, restart API + UI, hard-refresh 5173.
2. Align `backend\.env` with `.env.example` (`WHISPER_DEVICE=auto`,
   `WHISPER_COMPUTE_TYPE=auto`, `MBART_DEVICE=auto`).
3. If the Whisper banner persists: install `faster-whisper` in that Python and
   restart uvicorn (not only the frontend).
4. Name suggestions stay empty until meetings are saved with officer/attendee
   names.
5. Voice labels are not identities; one speaker may still split into two
   voices.
6. Optional: Python 3.12 + sentencepiece for mBART on Windows 3.13; Google
   Translate API for the Hiligaynon Google path.
7. `presiding_office` column/API leftovers may still exist on old databases;
   the form does not use them.
8. True client E2E encryption remains roadmap (`docs/ENCRYPTION_AT_REST.md`).

---

## 9. Decisions captured from discussion

- **No custom vocabulary editor.** Proper-noun bias is attendees + officer.
- **No spoken-language UI.** Always `auto` with Hiligaynon-biased prompts.
- **Voice N, not clock stamps,** in the meeting room and exports.
- **Voice 1 means most accurate transcription,** not “first speaker.”
- **GPU is opt-in via auto-detect on the API host,** with CPU INT8 fallback.
- **Do not run ASR in the cloud agent and expect the laptop GPU** to
  participate.

This file was written at the operator’s request to snapshot the session.
Update it when a later session lands a new stacked tip.
