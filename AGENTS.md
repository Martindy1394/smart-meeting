## Cursor Cloud specific instructions

- Product/services:
  - Backend API: FastAPI app in `backend` (see root `README.md` quick start).
  - Frontend: Vite React app in `frontend` (see root `README.md` quick start).
  - Redis/Postgres are optional for local smoke runs; backend degrades to local buffers/SQLite when they are unavailable.
- Use module invocations for Python tools in cloud VMs (`python3 -m uvicorn`, `python3 -m pytest`) because user-site script paths may not be on `PATH`.
- Standard run/build/test commands are documented in root `README.md` and `package.json` scripts; prefer those as source of truth.
- Non-obvious runtime caveat: on startup, backend may log Redis-unavailable warnings. This is expected in local no-Redis runs and does not block auth/meetings/API smoke tests.
- ML deps path: `requirements-ml.txt` lives under `backend/`, not the repo root. From `/workspace` install with `python3 -m pip install --user -r backend/requirements-ml.txt` (or `cd backend` first). This is optional and heavy; core app runs without it.
