[CmdletBinding()]
param(
    [switch]$StartMenu
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonw = Join-Path $repoRoot ".venv\Scripts\pythonw.exe"
if (-not (Test-Path -LiteralPath $pythonw -PathType Leaf)) {
    throw "Missing repository environment. Run setup_windows.ps1 first."
}

if ($StartMenu) {
    $shortcutFolder = [Environment]::GetFolderPath("Programs")
}
else {
    $shortcutFolder = [Environment]::GetFolderPath("Desktop")
}
if ([string]::IsNullOrWhiteSpace($shortcutFolder)) {
    throw "Windows could not resolve the requested known folder."
}

$shortcutPath = Join-Path $shortcutFolder "Defleco LAB.lnk"
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $pythonw
$shortcut.Arguments = "-m defleco_lab"
$shortcut.WorkingDirectory = $repoRoot
$shortcut.Description = "Defleco LAB"
$shortcut.Save()

if (-not (Test-Path -LiteralPath $shortcutPath -PathType Leaf)) {
    throw "Shortcut creation did not produce the expected file."
}
Write-Output $shortcutPath
