#!/usr/bin/env bash
# Launch YAUM via uv. Installs uv on first run; uv handles Python + deps.
#
# Usage:
#   ./run.sh                 # launch the Gradio UI
#   ./run.sh --server-name 0.0.0.0   # forward extra args straight to the app
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if ! command -v uv >/dev/null 2>&1; then
    echo "[yaum] uv not found — installing via https://astral.sh/uv/install.sh"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
    if ! command -v uv >/dev/null 2>&1; then
        echo "[yaum] uv install did not land in PATH. Open a new shell and retry." >&2
        exit 1
    fi
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[yaum] NVIDIA GPU detected. Using CUDA-enabled PyTorch from requirements.txt."
else
    echo "[yaum] No NVIDIA GPU detected. Warning: CUDA build will be used on CPU."
fi

# Ensure .venv exists
if [ ! -f ".venv/bin/python" ] && [ ! -f ".venv/Scripts/python.exe" ]; then
    echo "[yaum] Creating .venv with Python 3.12..."
    uv venv --python 3.12 .venv
fi

# Always sync dependencies
echo "[yaum] Syncing dependencies..."
uv pip install -r requirements.txt --index-strategy unsafe-best-match
if [ $? -ne 0 ]; then
    echo "[yaum] Dependency sync failed."
    exit 1
fi

echo "[yaum] Launching..."
if [ -f ".venv/bin/python" ]; then
    exec .venv/bin/python -m yaum.ui.app "$@"
else
    exec .venv/Scripts/python.exe -m yaum.ui.app "$@"
fi
