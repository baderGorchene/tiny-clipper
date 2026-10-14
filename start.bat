@echo off
rem Double-click to start Tiny Clipper. First run installs everything (a few minutes).
cd /d "%~dp0"
rem UTF-8 output, so log lines with emoji or any language never crash a job.
set PYTHONUTF8=1
if not exist .venv\Scripts\python.exe (
  echo Setting up Tiny Clipper for the first time...
  py -3 -m venv .venv 2>nul || python -m venv .venv
)
.venv\Scripts\python -m pip install -q --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
  echo Installing failed. Check your internet connection and try again.
  pause
  exit /b 1
)
echo Tiny Clipper is running at http://127.0.0.1:8765  -  keep this window open.
.venv\Scripts\python server.py
pause
