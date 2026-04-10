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

echo "[yaum] uv $(uv --version) — resolving environment..."
exec uv run \
    --python ">=3.10" \
    --with-requirements requirements.txt \
    python -m yaum.ui.app "$@"
