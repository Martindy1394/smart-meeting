@echo off
setlocal EnableExtensions
cd /d "%~dp0frontend"

if not exist "package.json" (
  echo Could not find frontend\package.json.
  echo Keep start-frontend.bat in the Technical repo root.
  pause
  exit /b 1
)

where npm >nul 2>&1
if errorlevel 1 (
  echo npm was not found. Install Node.js, then reopen this window.
  pause
  exit /b 1
)

if not exist "node_modules" (
  echo Installing frontend dependencies...
  call npm install
  if errorlevel 1 (
    echo npm install failed.
    pause
    exit /b 1
  )
)

title Smart Meeting UI
echo.
echo Starting UI at http://127.0.0.1:5173
echo The API must also be running (start-api.bat).
echo Leave this window open.
echo.

call npm run dev -- --host 0.0.0.0 --port 5173

echo.
echo Frontend stopped.
pause
