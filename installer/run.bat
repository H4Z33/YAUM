@echo off
REM Launch YAUM via uv on Windows. Installs uv on first run; uv handles
REM Python + dependencies. Works from cmd.exe and from Explorer double-click.
REM
REM Usage (from the repo root, or anywhere):
REM     installer\run.bat
REM     installer\run.bat --server-name 0.0.0.0

setlocal EnableExtensions EnableDelayedExpansion

REM ---- locate repo root (one level above this script) ----------------------
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%.." >nul
if errorlevel 1 (
    echo [yaum] could not cd to repo root from %SCRIPT_DIR%
    exit /b 1
)

REM ---- ensure uv is available ----------------------------------------------
where uv >nul 2>&1
if errorlevel 1 (
    echo [yaum] uv not found - installing via https://astral.sh/uv/install.ps1
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression"
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
    where uv >nul 2>&1
    if errorlevel 1 (
        echo [yaum] uv install did not land in PATH. Open a new shell and retry.
        popd >nul
        exit /b 1
    )
)

for /f "delims=" %%v in ('uv --version') do set "UV_VERSION=%%v"
echo [yaum] !UV_VERSION! - resolving environment...

uv run --python ">=3.10" --with-requirements requirements.txt python -m yaum.ui.app %*
set "EXIT_CODE=%ERRORLEVEL%"

popd >nul
endlocal & exit /b %EXIT_CODE%
