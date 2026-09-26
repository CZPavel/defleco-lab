[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"

Set-Location -LiteralPath $repoRoot

$dirty = & git status --porcelain
if ($LASTEXITCODE -ne 0) {
    throw "Git status failed."
}
if ($dirty) {
    throw "Local checkout has uncommitted changes. Commit/stash them before updating."
}

& git pull --ff-only origin main
if ($LASTEXITCODE -ne 0) {
    throw "git pull --ff-only failed."
}

if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    Write-Host "Local environment is missing; running first-time setup..."
    & (Join-Path $repoRoot "setup_windows.ps1")
    if ($LASTEXITCODE -ne 0) {
        throw "setup_windows.ps1 failed."
    }
}
else {
    # Editable install already points at this checkout. Re-install only to pick up
    # dependency/package metadata changes; this is intentionally lightweight.
    & $venvPython -m pip install -e ".[basler]"
    if ($LASTEXITCODE -ne 0) {
        throw "Editable install refresh failed."
    }
}

& $venvPython -c "import defleco_lab, cv2; from pypylon import pylon; print('Defleco LAB update OK; pylon ' + pylon.GetPylonVersionString())"
if ($LASTEXITCODE -ne 0) {
    throw "Post-update import check failed."
}

Write-Host "Defleco LAB is updated and ready. Use the existing Desktop shortcut or Start Defleco LAB.cmd." -ForegroundColor Green
