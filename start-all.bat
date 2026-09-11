@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not exist "start-api.bat" (
  echo Keep start-all.bat next to start-api.bat and start-frontend.bat.
  pause
  exit /b 1
)

echo Opening API and UI in two windows...
start "Smart Meeting API" cmd /k "%~dp0start-api.bat"
timeout /t 2 /nobreak >nul
start "Smart Meeting UI" cmd /k "%~dp0start-frontend.bat"
echo.
echo API:  http://127.0.0.1:8000
echo App:  http://127.0.0.1:5173
echo.
echo You can close this window. Leave the two new windows open.
timeout /t 4 /nobreak >nul
