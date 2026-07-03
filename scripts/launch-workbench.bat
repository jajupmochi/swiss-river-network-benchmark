@echo off
REM One-click launcher for the Swiss River Network Benchmark workbench (Windows).
REM
REM For non-technical users: double-click this file. It installs everything on the
REM first run and opens the app in your web browser. No coding required.
setlocal
cd /d "%~dp0.."

where uv >nul 2>nul
if errorlevel 1 (
  echo First run: installing the uv package manager ^(one-time^)...
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)

echo Starting the workbench - a browser tab will open at http://127.0.0.1:7860
echo (the first run downloads dependencies and can take a few minutes). Close this window to stop.
set SRN_INBROWSER=1
set SRN_HOST=127.0.0.1
uv run --extra app python -m swissrivernetwork.app.workbench
pause
