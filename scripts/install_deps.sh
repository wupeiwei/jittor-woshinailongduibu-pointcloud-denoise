#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"
source scripts/env.sh

REQ="${REQ:-requirements.txt}"
WHEELHOUSE="${WHEELHOUSE:-}"
PIP_INDEX_URL="${PIP_INDEX_URL:-}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-}"

PIP_ARGS=()
if [ -n "$PIP_INDEX_URL" ]; then
  PIP_ARGS+=(-i "$PIP_INDEX_URL")
fi
if [ -n "$PIP_TRUSTED_HOST" ]; then
  PIP_ARGS+=(--trusted-host "$PIP_TRUSTED_HOST")
fi

"$PYTHON" -m pip --version >/dev/null
"$PYTHON" -m pip install "${PIP_ARGS[@]}" --upgrade pip setuptools wheel

if [ -n "$WHEELHOUSE" ]; then
  echo "Installing dependencies from local wheelhouse: $WHEELHOUSE"
  "$PYTHON" -m pip install --no-index --find-links "$WHEELHOUSE" -r "$REQ"
else
  if [ -n "$PIP_INDEX_URL" ]; then
    echo "Installing dependencies with PIP_INDEX_URL=$PIP_INDEX_URL"
  fi
  "$PYTHON" -m pip install "${PIP_ARGS[@]}" -r "$REQ"
fi

# Be explicit: CUDA Jittor training on RTX 5060 Ti / A6000 needs CuPy.
if ! "$PYTHON" - <<'PY' >/dev/null 2>&1
import cupy
PY
then
  echo "CuPy not importable; installing CUDA 12.x wheel used by RTX 5060 Ti / A6000 environments."
  "$PYTHON" -m pip install "${PIP_ARGS[@]}" "cupy-cuda12x>=13.0,<14.0"
fi

"$PYTHON" scripts/check_env.py
