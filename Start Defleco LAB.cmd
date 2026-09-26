@echo off
setlocal
set "REPO_ROOT=%~dp0"
set "PYTHONW=%REPO_ROOT%.venv\Scripts\pythonw.exe"
cd /d "%REPO_ROOT%"

if not exist "%PYTHONW%" (
    echo Defleco LAB local environment is missing.
    echo Run: powershell -NoProfile -ExecutionPolicy Bypass -File setup_windows.ps1
    pause
    exit /b 1
)

start "" "%PYTHONW%" -m defleco_lab
exit /b 0
