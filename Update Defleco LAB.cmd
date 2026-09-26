@echo off
setlocal
set "REPO_ROOT=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%REPO_ROOT%tools\update_local_windows.ps1"
set "APP_EXIT=%ERRORLEVEL%"
if not "%APP_EXIT%"=="0" (
    echo.
    echo Update failed with error code %APP_EXIT%.
    pause
)
exit /b %APP_EXIT%
