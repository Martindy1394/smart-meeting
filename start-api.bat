@echo off
setlocal EnableExtensions
cd /d "%~dp0backend"

if not exist "app\main.py" (
  echo Could not find backend\app\main.py.
  echo Keep start-api.bat in the repo root.
  pause
  exit /b 1
)

if not exist ".env" if exist ".env.example" (
  copy /Y ".env.example" ".env" >nul
  echo Created backend\.env from .env.example
)

set "PYEXE="
if exist ".venv\Scripts\python.exe" set "PYEXE=.venv\Scripts\python.exe"
if not defined PYEXE if exist "..\.venv\Scripts\python.exe" set "PYEXE=..\.venv\Scripts\python.exe"

title Smart Meeting API
echo.
echo Starting API at http://127.0.0.1:8000
echo Health: http://127.0.0.1:8000/api/health
echo Docs:   http://127.0.0.1:8000/docs
echo Leave this window open.
echo.

echo Checking faster-whisper in this Python...
if defined PYEXE (
  "%PYEXE%" "scripts\ensure_faster_whisper.py"
  if errorlevel 1 (
    echo Failed to install faster-whisper into "%PYEXE%".
    pause
    exit /b 1
  )
  echo.
  "%PYEXE%" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir app
) else (
  py -3 "scripts\ensure_faster_whisper.py"
  if errorlevel 1 (
    echo Failed to install faster-whisper. Install Python 3 and retry.
    pause
    exit /b 1
  )
  echo.
  py -3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir app
)

echo.
echo API stopped.
pause
