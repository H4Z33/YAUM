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

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
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

Write-Host "[yaum] uv $(uv --version) - resolving environment..."
uv run `
    --python ">=3.10" `
    --with-requirements requirements.txt `
    python -m yaum.ui.app @args
exit $LASTEXITCODE
