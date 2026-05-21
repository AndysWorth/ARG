#!/usr/bin/env bash
# ARG bootstrap — one-time setup. The ONLY step in ARG that may touch the network.
#
# Idempotent: skips work that is already done. Safe to re-run.
#
#   - Verifies Python 3.11+ (stdlib `venv` only — no Poetry/Conda/uv)
#   - Creates / reuses `.venv/`
#   - Installs project via `pip install -e .[dev]`, preferring `./vendor/` wheel cache
#   - Installs Tesseract via Homebrew (provides tessdata for pymupdf OCR)
#   - Installs Ollama via Homebrew if absent
#   - Pulls qwen3.6:35b-a3b-q4_K_M and nomic-embed-text ONLY if absent
#   - Downloads D3.js v7 once to arg/static/d3.min.js (served locally, never CDN)
#
# After bootstrap completes, ARG makes zero outbound network calls.

set -euo pipefail

# Resolve project root (parent of scripts/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

log()  { printf '\033[1;36m[bootstrap]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[bootstrap]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[bootstrap]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. Python 3.11+
# ---------------------------------------------------------------------------
if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="python3.11"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    die "Python 3 not found. Install via 'brew install python@3.11'."
fi

PY_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
PY_MAJOR="$("${PYTHON_BIN}" -c 'import sys; print(sys.version_info[0])')"
PY_MINOR="$("${PYTHON_BIN}" -c 'import sys; print(sys.version_info[1])')"
if [ "${PY_MAJOR}" -lt 3 ] || { [ "${PY_MAJOR}" -eq 3 ] && [ "${PY_MINOR}" -lt 11 ]; }; then
    die "Python 3.11+ required (found ${PY_VERSION}). Install via 'brew install python@3.11'."
fi
log "Using ${PYTHON_BIN} (${PY_VERSION})."

# ---------------------------------------------------------------------------
# 2. Homebrew (required for tesseract + ollama)
# ---------------------------------------------------------------------------
if ! command -v brew >/dev/null 2>&1; then
    die "Homebrew not found. Install from https://brew.sh and re-run bootstrap."
fi

# ---------------------------------------------------------------------------
# 3. Tesseract (tessdata for pymupdf OCR)
# ---------------------------------------------------------------------------
if brew list --formula 2>/dev/null | grep -qx tesseract; then
    log "tesseract already installed."
else
    log "Installing tesseract via Homebrew..."
    brew install tesseract
fi

# ---------------------------------------------------------------------------
# 4. Ollama
# ---------------------------------------------------------------------------
if ! command -v ollama >/dev/null 2>&1; then
    log "Installing ollama via Homebrew..."
    brew install ollama
else
    log "ollama already installed."
fi

# Start the daemon if it's not already responding on localhost:11434.
if ! curl -fsS http://localhost:11434/api/version >/dev/null 2>&1; then
    log "Starting ollama service (brew services)..."
    brew services start ollama >/dev/null 2>&1 || nohup ollama serve >/dev/null 2>&1 &
    # Wait up to 30s for the daemon to come up.
    for _ in $(seq 1 30); do
        if curl -fsS http://localhost:11434/api/version >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done
    curl -fsS http://localhost:11434/api/version >/dev/null 2>&1 \
        || die "ollama daemon did not start on localhost:11434."
fi
log "ollama is reachable at localhost:11434."

# Pull required models only if absent. `ollama list` lines look like:
#   NAME                            ID    SIZE   MODIFIED
present_models="$(ollama list 2>/dev/null | awk 'NR>1 {print $1}')"

want_llm="qwen3.6:35b-a3b-q4_K_M"
want_embed="nomic-embed-text"

if printf '%s\n' "${present_models}" | grep -Fxq "${want_llm}"; then
    log "model already present: ${want_llm}"
else
    log "Pulling ${want_llm} (this takes a while; ~38GB)..."
    ollama pull "${want_llm}"
fi

# `nomic-embed-text` may be listed as `nomic-embed-text:latest`; accept either.
if printf '%s\n' "${present_models}" | grep -Eq "^${want_embed}(:|$)"; then
    log "model already present: ${want_embed}"
else
    log "Pulling ${want_embed}..."
    ollama pull "${want_embed}"
fi

# ---------------------------------------------------------------------------
# 5. Python virtual environment
# ---------------------------------------------------------------------------
if [ ! -d ".venv" ]; then
    log "Creating .venv via ${PYTHON_BIN} -m venv ..."
    "${PYTHON_BIN}" -m venv .venv
else
    log ".venv already exists; reusing."
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip >/dev/null

if [ -d "vendor" ] && [ -n "$(ls -A vendor 2>/dev/null)" ]; then
    log "Installing project from ./vendor/ wheel cache (offline)..."
    pip install --no-index --find-links ./vendor/ -e ".[dev]"
else
    log "Installing project from PyPI (no ./vendor/ cache present)..."
    log "Tip: pre-download wheels with 'pip download -e . -d ./vendor/' for offline installs."
    pip install -e ".[dev]"
fi

# ---------------------------------------------------------------------------
# 6. D3.js (served locally; never loaded from a CDN at runtime)
# ---------------------------------------------------------------------------
D3_PATH="arg/static/d3.min.js"
if [ -f "${D3_PATH}" ]; then
    log "${D3_PATH} already present."
else
    mkdir -p "$(dirname "${D3_PATH}")"
    log "Downloading D3.js v7 to ${D3_PATH}..."
    curl -fsSL "https://d3js.org/d3.v7.min.js" -o "${D3_PATH}"
fi

# ---------------------------------------------------------------------------
# 7. Pre-commit hooks (idempotent)
# ---------------------------------------------------------------------------
if command -v pre-commit >/dev/null 2>&1; then
    log "Installing pre-commit git hook..."
    pre-commit install >/dev/null
fi

log "Bootstrap complete. Activate the environment with: source .venv/bin/activate"
