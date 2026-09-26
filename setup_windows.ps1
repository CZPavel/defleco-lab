[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvRoot = Join-Path $repoRoot ".venv"
$venvPython = Join-Path $venvRoot "Scripts\python.exe"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Program,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $Program $($Arguments -join ' ')"
    }
}

function Invoke-PythonCode {
    param(
        [Parameter(Mandatory = $true)][string]$Python,
        [Parameter(Mandatory = $true)][string]$Code
    )
    $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($Code))
    $bootstrap = "import base64; exec(base64.b64decode('$encoded'))"
    Invoke-Checked -Program $Python -Arguments @("-c", $bootstrap)
}

try {
    Set-Location -LiteralPath $repoRoot
    if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
        $pyLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
        if ($null -eq $pyLauncher) {
            throw "Python Launcher was not found. Install 64-bit Python 3.11, then run this script again."
        }
        Write-Host "Creating repository environment with Python 3.11..."
        Invoke-Checked -Program $pyLauncher.Source -Arguments @("-3.11", "-m", "venv", $venvRoot)
    }

    $runtimeInfo = @(Invoke-PythonCode -Python $venvPython -Code @'
import platform
import struct
print(platform.python_version())
print(struct.calcsize("P") * 8)
'@)
    $version = $runtimeInfo[0]
    $bits = $runtimeInfo[1]
    if (-not ($version -like "3.11.*") -or $bits -ne "64") {
        throw "Existing .venv uses Python $version ($bits-bit); Python 3.11 x64 is required."
    }

    Write-Host "Updating packaging tools..."
    Invoke-Checked -Program $venvPython -Arguments @(
        "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"
    )
    Write-Host "Installing Defleco LAB with Basler support in editable mode..."
    Invoke-Checked -Program $venvPython -Arguments @("-m", "pip", "install", "-e", ".[basler]")

    $importCheck = @'
import cv2
import numpy
import PySide6
import pypylon
import defleco_lab
from pypylon import pylon
print("Imports OK")
print("pylon Runtime " + pylon.GetPylonVersionString())
'@
    Invoke-PythonCode -Python $venvPython -Code $importCheck

    $smokeCheck = @'
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PySide6.QtWidgets import QApplication
from defleco_lab.app import configure_application
from defleco_lab.gui.main_window import MainWindow
from defleco_lab.processing import load_builtin_methods
app = QApplication([])
configure_application(app)
load_builtin_methods()
window = MainWindow()
app.processEvents()
assert window.last_original is not None
window.close()
app.processEvents()
assert not window.processor.isRunning()
print("Synthetic GUI smoke test OK")
'@
    Invoke-PythonCode -Python $venvPython -Code $smokeCheck
    Write-Host "Defleco LAB setup completed successfully." -ForegroundColor Green
    Write-Host "Start it with: Start Defleco LAB.cmd"
}
catch {
    Write-Host "Setup failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "If pypylon cannot load, install the compatible Basler pylon Runtime separately." -ForegroundColor Yellow
    exit 1
}
