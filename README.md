# Smart Meeting — Online Minute-Making

A full-stack platform that captures live meeting audio, transcribes it in
real time with **Whisper** (two-pass pipeline, explicit **Hiligaynon** support),
condenses it with **BART** summarization (bullet points *or* numbered
paragraphs), and translates it on demand with **mBART** into 14+ languages — all
behind a secure JWT authentication system with per-user meeting history.

```
┌──────────────┐   PCM/WS    ┌────────────────────────────┐
│  React (Vite)│ ──────────► │  FastAPI backend           │
│  AudioWorklet│             │  • Auth (JWT + bcrypt)     │
│  16kHz PCM   │ ◄────────── │  • WebSocket live captions  │
│  transcript/ │  live +     │  • Whisper 2-pass           │
│  summary/    │  final      │  • InvokeLLM: BART / mBART  │
│  translate   │             │  • Meetings CRUD (SQLite/PG)│
└──────────────┘             │  • Redis audio memory store │
                             └─────────────┬──────────────┘
                                           │
                                      ┌────▼────┐
                                      │  Redis  │
                                      │ PCM/WAV │
                                      └─────────┘
```

## Features

- **Redis audio memory** — every recorded PCM chunk is appended to Redis during
  capture; finalized WAV is cached in Redis too (disk archive kept for playback).
  Live caption offsets resume from Redis on reconnect.
- **Multi-hour board meetings** — live transcription is designed for 8h+ sessions
  (default soft cap 16h): 48h Redis TTL, WebSocket keepalives, chunked final
  Whisper ASR, and caption-merge optimizations so long captions stay fast.
- **Authentication** — signup with email + strong-password validation, login,
  JWT (7-day expiry), bcrypt hashing (12 rounds), protected routes, logout.
- **Live transcription** — raw 16 kHz mono PCM captured with an `AudioWorklet`
  (no MediaRecorder / WebM decoding), streamed over WebSockets, decoded live
  with a fast Whisper model for word-by-word captions.
- **Two-pass accuracy** — when you stop recording the server re-transcribes the
  whole recording with a stronger Whisper pass (fine-tuned Hiligaynon/PH model
  when available, else faster-whisper `medium`) and replaces live captions with
  the finalized transcript (**Refined** badge).
- **Automatic language (Hiligaynon-biased)** — no Spoken language picker.
  Meetings use `auto`; ASR biases Hiligaynon via `WHISPER_DEFAULT_LANGUAGE=hil`
  and prefers the Philippine dialect Whisper checkpoint
  (`rbcurzon/whisper-medium-ph`, Visayan-aware) over stock Whisper, with
  loudness normalization and short prompts (long word-list prompts were being
  echoed). Whisper has no native `hil` token, so Hiligaynon uses **auto-detect**
  (never forced Tagalog `tl`) plus Hiligaynon prompts / PH-medium models.
  For best accuracy, fine-tune on the **UP-DSP Philippine Languages Database
  (PLD)** Hiligaynon subset (~41h) with
  [`scripts/hiligaynon_asr/import_pld.py`](scripts/hiligaynon_asr/import_pld.py)
  and set `WHISPER_HILIGAYNON_FINE_TUNED_MODEL`
  (see [`docs/PLD.md`](docs/PLD.md) and
  [`docs/FINE_TUNE_HILIGAYNON.md`](docs/FINE_TUNE_HILIGAYNON.md)).
- **Optional RNN-T live captions** — when NeMo is installed
  (`requirements-rnnt.txt`), live PH captions can use a Tagalog
  FastConformer hybrid RNN-T for lower latency; final pass stays on Whisper
  (see [`docs/RNN_T_LIVE.md`](docs/RNN_T_LIVE.md)).
- **English meeting summaries** — mBART translates the **full** transcript into
  English with sliding context windows (PH/mixed via `id_ID`), then topic-aware
  BART summarizes with topic overlap + coverage restore into meeting-minutes
  sections (Discussion / Decisions / Action items).
- **mBART translation** — context-aware English of the whole transcript, plus
  Spanish, French, German, Italian, Portuguese, Arabic, Hindi, Japanese,
  Chinese, Russian, Dutch, Korean, Hiligaynon, Tagalog.
- **Meeting details** — capture each meeting's title, venue, date & time, and
  attendee list; shown and editable in the meeting room and surfaced in history.
- **Dashboard** — overview of meeting counts (total, this week, transcript /
  translation / audio) and a feed of the latest mBART translations.
- **Meeting history** — sidebar with search, load, rename, and delete.
- **Production concerns** — CORS, rate limiting, input validation/sanitization,
  graceful error handling, reconnection with exponential backoff, loading
  states, responsive (desktop + mobile) UI.

## Project layout

```
backend/         FastAPI application
  app/
    main.py           app wiring, CORS, health, error handling
    config.py         env-driven settings
    database.py       SQLAlchemy engine/session
    models.py         User / Meeting / TranscriptSegment
    schemas.py        Pydantic request/response + validation
    security.py       bcrypt + JWT
    deps.py           auth dependencies
    limiter.py        in-memory rate limiter dependency
    languages.py      supported translation languages
    routers/          auth, meetings, ai
    services/         transcription (Whisper), llm (BART/mBART), audio
    ws/               WebSocket live-transcription endpoint
  requirements.txt      core deps (always installable)
  requirements-ml.txt   heavy ML deps (Whisper / transformers / torch)
frontend/        React + Vite SPA
  public/pcm-worklet.js  AudioWorklet PCM downsampler
  src/                   pages, components, hooks, api client, auth context
```

## Quick start (development)

### 1. Backend

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate   # or use your env manager
pip install -r requirements.txt
# Optional (enables real Whisper + BART + mBART; multi-GB downloads):
# pip install -r requirements-ml.txt
cp .env.example .env        # then edit JWT_SECRET_KEY etc.
uvicorn app.main:app --reload --port 8000
```

The API is served at `http://127.0.0.1:8000` (`/api/health` for a status probe,
`/docs` for interactive OpenAPI docs).

### 2. Frontend

```bash
cd frontend
npm install
npm run dev      # http://localhost:5173  (proxies /api and /ws to :8000)
```

Open http://localhost:5173, create an account, click **New meeting**, and press
**Start recording**.

> **Note on ML dependencies:** the core app runs without the heavy ML packages
> so you can exercise auth, meeting management, the UI, and the WebSocket
> pipeline immediately. Summarization uses a lightweight extractive fallback
> until `facebook/bart-large-cnn` is available; translation requires
> `requirements-ml.txt` (mBART). Install `requirements-ml.txt` and set the
> Whisper model sizes in `.env` to enable the full high-accuracy pipeline.

## The `InvokeLLM` / Whisper ASR integrations

All AI features flow through single, consistent integration surfaces:

- `app/services/llm.py :: invoke_llm(task, text, **kwargs)` — `task="summarize"`
  routes to BART; `task="translate"` routes to mBART.
- `app/services/asr.py` — **Whisper ASR** for all audio → text processing:
  - `transcribe_pcm(..., live=True|False)` for live captions / full pass
  - `transcribe_file(path)` for saved or uploaded WAV recordings
  - `persist_transcript(...)` writes segments onto the meeting
- `app/services/transcription.py` — Whisper backends used by the ASR facade:
  - **Live:** faster-whisper with **10s windows overlapping by 5s**, decode
    language `tl` for PH meetings (optional CT2 via
    `WHISPER_LIVE_TAGALOG_MODEL` / `WHISPER_LIVE_HILIGAYNON_MODEL`)
  - **Final (Stop Recording):** `WHISPER_FINAL_BACKEND=auto` prefers language
    HF candidates — Tagalog: custom → `WHISPER_TAGALOG_MODEL` → PH medium;
    Hiligaynon: custom → `WHISPER_HILIGAYNON_MODEL` — then faster-whisper

## Configuration

See `backend/.env.example`. Key variables:

| Variable | Default | Purpose |
|---|---|---|
| `JWT_SECRET_KEY` | dev value | **Change in production.** |
| `DATABASE_URL` | `sqlite:///./smart_meeting.db` | Use a PostgreSQL DSN in prod. |
| `WHISPER_LIVE_MODEL` | `small` | Fast live-caption Whisper model. |
| `WHISPER_FINAL_MODEL` | `medium` | faster-whisper fallback if HF final model fails. |
| `WHISPER_FINAL_BACKEND` | `auto` | `auto` prefers HF Tagalog/Hiligaynon candidates, then FW. |
| `WHISPER_HILIGAYNON_FINE_TUNED_MODEL` | _(empty)_ | Your Hiligaynon fine-tune (HF/local); tried first. |
| `WHISPER_HILIGAYNON_MODEL` | `rbcurzon/whisper-medium-ph` | Philippine HF fallback for Stop Recording. |
| `WHISPER_LIVE_HILIGAYNON_MODEL` | _(empty)_ | Optional CT2 fine-tune for live captions. |
| `WHISPER_TAGALOG_FINE_TUNED_MODEL` | _(empty)_ | Your Tagalog fine-tune (HF/local); tried first. |
| `WHISPER_TAGALOG_MODEL` | `LWobole/whisper-small-tagalog` | Tagalog HF model for `tl` meetings. |
| `WHISPER_LIVE_TAGALOG_MODEL` | _(empty)_ | Optional CT2 Tagalog fine-tune for live captions. |
| `WHISPER_TAGALOG_FINAL_LANGUAGE_MODE` | `prefer_forced` | Force `tl` then auto retry for code-switch. |
| `WHISPER_LIVE_WINDOW_SECONDS` | `10.0` | Live ASR window length. |
| `WHISPER_LIVE_HOP_SECONDS` | `5.0` | Live ASR hop (10s window overlapping by 5s). |
| `WHISPER_DEFAULT_LANGUAGE` | `auto` | Whisper detects language (no UI picker). |
| `REDIS_URL` | `redis://127.0.0.1:6379/0` | Redis memory store for recorded PCM/WAV. |
| `REDIS_AUDIO_TTL_SECONDS` | `172800` | TTL for Redis audio keys (48h; 0 = no expiry). |
| `MAX_MEETING_HOURS` | `16` | Soft cap for one live recording (board meetings). |
| `WHISPER_FINAL_CHUNK_SECONDS` | `600` | Final ASR chunk size for multi-hour audio. |
| `WS_KEEPALIVE_SECONDS` | `25` | WebSocket ping interval for long sessions. |
| `BART_MODEL` | `facebook/bart-large-cnn` | Summarization. |
| `MBART_MODEL` | `facebook/mbart-large-50-many-to-many-mmt` | Translation. |
| `ALLOW_LLM_FALLBACK` | `true` | Enable extractive summary fallback. |
| `BART_MAX_INPUT_TOKENS` | `960` | Max tokens per topic chunk (under BART’s 1024 limit). |
| `BART_TOPIC_SIMILARITY_THRESHOLD` | `0.22` | Split topics when consecutive-unit TF cosine falls below this. |
| `BART_TOPIC_MIN_UNITS` | `2` | Min discourse units before a similarity split. |
| `CORS_ORIGINS` | localhost dev | Comma-separated allowed origins. |

## API summary

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/api/auth/signup` | – | Create account, returns JWT |
| POST | `/api/auth/login` | – | Login, returns JWT |
| GET | `/api/auth/me` | ✓ | Current user |
| GET | `/api/meetings` | ✓ | List (supports `?search=`) |
| POST | `/api/meetings` | ✓ | Create (title, venue, date, attendees) |
| GET | `/api/meetings/{id}` | ✓ | Detail (details + transcript/summary/translation) |
| PATCH | `/api/meetings/{id}` | ✓ | Update title / venue / date / attendees |
| DELETE | `/api/meetings/{id}` | ✓ | Delete |
| GET | `/api/meetings/{id}/audio` | ✓ | Stream saved WAV for playback |
| POST | `/api/meetings/{id}/audio` | ✓ | Upload WAV/PCM (+ Whisper ASR) |
| POST | `/api/meetings/{id}/retranscribe` | ✓ | Re-run Whisper ASR on saved audio |
| GET | `/api/ai/languages` | ✓ | Supported languages |
| POST | `/api/ai/summarize` | ✓ | BART summary (`bullets`/`numbered`) |
| POST | `/api/ai/translate` | ✓ | mBART translation |
| WS | `/ws/transcribe?token=&meeting_id=` | ✓ | Live PCM → Whisper ASR + finalize |

## Security notes

- Passwords hashed with bcrypt (rounds=12); never stored or logged in plaintext.
- JWTs signed with `HS256`; set a strong `JWT_SECRET_KEY` and serve over HTTPS
  in production (terminate TLS at your proxy / load balancer).
- Per-IP rate limiting on auth and AI endpoints (in-memory; back with Redis for
  multi-process deployments).
- All meeting data is scoped to the authenticated owner.
