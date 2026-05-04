# Launch YAUM via uv on Windows. Installs uv on first run.
#
# Usage (from PowerShell):
#     .\run.ps1
#     .\run.ps1 --server-name 0.0.0.0   # extra args are forwarded to the app
#
# If PowerShell refuses to execute this file, run once:
#     Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
#Requires -Version 5.1
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "[yaum] uv not found - installing via https://astral.sh/uv/install.ps1"
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Error "[yaum] uv install did not land in PATH. Open a new shell and retry."
        exit 1
    }
}

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host "[yaum] NVIDIA GPU detected. Using CUDA-enabled PyTorch from requirements.txt."
} else {
    Write-Host "[yaum] No NVIDIA GPU detected. Warning: CUDA build will be used on CPU."
}

# Ensure .venv exists
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "[yaum] Creating .venv with Python 3.12..."
    uv venv --python 3.12 .venv
}

# Always sync dependencies
Write-Host "[yaum] Syncing dependencies..."
uv pip install -r requirements.txt --index-strategy unsafe-best-match
if ($LASTEXITCODE -ne 0) {
    Write-Error "[yaum] Dependency sync failed."
    exit $LASTEXITCODE
}

Write-Host "[yaum] Launching..."
& .venv\Scripts\python.exe -m yaum.ui.app @args
exit $LASTEXITCODE
