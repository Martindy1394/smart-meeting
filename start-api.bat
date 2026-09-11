@echo off
setlocal EnableExtensions
cd /d "%~dp0backend"

if not exist "app\main.py" (
  echo Could not find backend\app\main.py.
  echo Keep start-api.bat in the Technical repo root.
  pause
  exit /b 1
)

if not exist ".env" if exist ".env.example" (
  copy /Y ".env.example" ".env" >nul
  echo Created backend\.env from .env.example
)

title Smart Meeting API
echo.
echo Starting API at http://127.0.0.1:8000
echo Health: http://127.0.0.1:8000/api/health
echo Docs:   http://127.0.0.1:8000/docs
echo Leave this window open.
echo.

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir app
) else if exist "..\.venv\Scripts\python.exe" (
  "..\.venv\Scripts\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir app
) else (
  py -3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir app
)

echo.
echo API stopped.
pause
