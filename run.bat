@echo off
setlocal

:: Launch YAUM via the local .venv on Windows.
:: On first run (or after deleting .venv), installs all dependencies including
:: the CUDA-enabled PyTorch from the PyTorch wheel server.
::
:: Usage:
::     run.bat
::     run.bat --server-name 0.0.0.0   # extra args are forwarded to the app

:: Change to the directory where this script is located
cd /d "%~dp0"

:: ── 1. Ensure uv is available ────────────────────────────────────────────────
where uv >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [yaum] uv not found - installing via https://astral.sh/uv/install.ps1
    powershell -ExecutionPolicy ByPass -Command "Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression"
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
    where uv >nul 2>nul
    if %ERRORLEVEL% neq 0 (
        echo [yaum] uv install did not land in PATH. Please open a new shell and retry.
        exit /b 1
    )
)

:: ── 2. Ensure .venv is synced ────────────────────────────────────────────────
if not exist ".venv\Scripts\python.exe" (
    echo [yaum] Creating .venv with Python 3.12...
    uv venv --python 3.12 .venv
    if %ERRORLEVEL% neq 0 ( echo [yaum] Failed to create .venv. & exit /b 1 )
)

:: Always run sync to ensure dependencies are met. uv is fast.
echo [yaum] Syncing dependencies...
uv pip install -r requirements.txt --index-strategy unsafe-best-match
if %ERRORLEVEL% neq 0 (
    echo [yaum] Dependency sync failed.
    exit /b 1
)

:: ── 3. Report GPU status ────────────────────────────────────────────────────
where nvidia-smi >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [yaum] NVIDIA GPU detected. Using CUDA-enabled PyTorch from requirements.txt.
) else (
    echo [yaum] No NVIDIA GPU detected. Warning: CUDA build will be used on CPU.
)

:: ── 4. Launch using the local .venv directly ────────────────────────────────
if not defined TORCH_SHOW_CPP_STACKTRACES set "TORCH_SHOW_CPP_STACKTRACES=1"
echo [yaum] CUDA diagnostics: TORCH_SHOW_CPP_STACKTRACES=%TORCH_SHOW_CPP_STACKTRACES%

echo [yaum] Launching...
.venv\Scripts\python.exe -m yaum.ui.app %*

if %ERRORLEVEL% neq 0 (
    echo [yaum] app exited with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

endlocal
