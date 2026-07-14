# Smart Meeting — Online Minute-Making

A full-stack platform that captures live meeting audio, transcribes it in
real time with **Whisper** (two-pass pipeline, explicit **Hiligaynon** support),
condenses it with **BART** summarization (bullet points *or* numbered
paragraphs), and translates it on demand with **mBART** into 14+ languages — all
behind a secure JWT authentication system with per-user meeting history.

```
┌──────────────┐   PCM/WS    ┌───────────────────────────┐
│  React (Vite)│ ──────────► │  FastAPI backend          │
│  AudioWorklet│             │  • Auth (JWT + bcrypt)    │
│  16kHz PCM   │ ◄────────── │  • WebSocket live captions │
│  transcript/ │  live +     │  • Whisper 2-pass          │
│  summary/    │  final      │  • InvokeLLM: BART / mBART  │
│  translate   │             │  • Meetings CRUD (SQLite/PG)│
└──────────────┘             └───────────────────────────┘
```

## Features

- **Authentication** — signup with email + strong-password validation, login,
  JWT (7-day expiry), bcrypt hashing (12 rounds), protected routes, logout.
- **Live transcription** — raw 16 kHz mono PCM captured with an `AudioWorklet`
  (no MediaRecorder / WebM decoding), streamed over WebSockets, decoded live
  with a fast Whisper model for word-by-word captions.
- **Two-pass accuracy** — when you stop recording the server re-transcribes the
  whole recording with `large-v3` (`beam_size=5`, `best_of=5`, VAD filtering,
  `condition_on_previous_text=True`) and replaces the live captions with the
  finalized transcript, flagged with a **Refined** badge.
- **Hiligaynon dialect** — language pinned to `hil` (falls back to the closest
  supported Philippine language if a Whisper build lacks the token, logged).
- **BART summarization** — one click, two formats: bullet points or numbered
  sentences within paragraphs.
- **mBART translation** — translate the finalized transcript into Spanish,
  French, German, Italian, Portuguese, Arabic, Hindi, Japanese, Chinese,
  Russian, Dutch, Korean, Hiligaynon, Tagalog (+ English).
- **Meeting details** — capture each meeting's title, venue, date & time, and
  attendee list; shown and editable in the meeting room and surfaced in history.
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

## The `InvokeLLM` / `TranscribeAudio` integrations

All AI features flow through single, consistent integration surfaces:

- `app/services/llm.py :: invoke_llm(task, text, **kwargs)` — `task="summarize"`
  routes to BART; `task="translate"` routes to mBART.
- `app/services/transcription.py :: transcribe_live()` / `transcribe_final()` —
  the fast and full-accuracy Whisper passes.

## Configuration

See `backend/.env.example`. Key variables:

| Variable | Default | Purpose |
|---|---|---|
| `JWT_SECRET_KEY` | dev value | **Change in production.** |
| `DATABASE_URL` | `sqlite:///./smart_meeting.db` | Use a PostgreSQL DSN in prod. |
| `WHISPER_LIVE_MODEL` | `base` | Fast live-caption model. |
| `WHISPER_FINAL_MODEL` | `large-v3` | Full-accuracy finalization model. |
| `WHISPER_DEFAULT_LANGUAGE` | `hil` | Hiligaynon. |
| `BART_MODEL` | `facebook/bart-large-cnn` | Summarization. |
| `MBART_MODEL` | `facebook/mbart-large-50-many-to-many-mmt` | Translation. |
| `ALLOW_LLM_FALLBACK` | `true` | Enable extractive summary fallback. |
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
| GET | `/api/ai/languages` | ✓ | Supported languages |
| POST | `/api/ai/summarize` | ✓ | BART summary (`bullets`/`numbered`) |
| POST | `/api/ai/translate` | ✓ | mBART translation |
| WS | `/ws/transcribe?token=&meeting_id=` | ✓ | Live PCM stream + finalization |

## Security notes

- Passwords hashed with bcrypt (rounds=12); never stored or logged in plaintext.
- JWTs signed with `HS256`; set a strong `JWT_SECRET_KEY` and serve over HTTPS
  in production (terminate TLS at your proxy / load balancer).
- Per-IP rate limiting on auth and AI endpoints (in-memory; back with Redis for
  multi-process deployments).
- All meeting data is scoped to the authenticated owner.
