@echo off
setlocal
set "REPO_ROOT=%~dp0"
set "PYTHON=%REPO_ROOT%.venv\Scripts\python.exe"
cd /d "%REPO_ROOT%"

if not exist "%PYTHON%" (
    echo Defleco LAB local environment is missing.
    echo Run: powershell -NoProfile -ExecutionPolicy Bypass -File setup_windows.ps1
    pause
    exit /b 1
)

"%PYTHON%" -m defleco_lab
set "APP_EXIT=%ERRORLEVEL%"
if not "%APP_EXIT%"=="0" (
    echo.
    echo Defleco LAB exited with error code %APP_EXIT%.
    pause
)
exit /b %APP_EXIT%
