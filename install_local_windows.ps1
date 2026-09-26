[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $repoRoot "setup_windows.ps1")
if ($LASTEXITCODE -ne 0) {
    throw "Environment setup failed with exit code $LASTEXITCODE."
}
$shortcut = & (Join-Path $repoRoot "tools\create_windows_shortcut.ps1")
Write-Host "Local installation is ready." -ForegroundColor Green
Write-Host "Desktop shortcut: $shortcut"
