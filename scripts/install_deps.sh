#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"
source scripts/env.sh

REQ="${REQ:-requirements.txt}"
WHEELHOUSE="${WHEELHOUSE:-}"
PIP_INDEX_URL="${PIP_INDEX_URL:-}"

"$PYTHON" -m pip --version >/dev/null
"$PYTHON" -m pip install --upgrade pip setuptools wheel

if [ -n "$WHEELHOUSE" ]; then
  echo "Installing dependencies from local wheelhouse: $WHEELHOUSE"
  "$PYTHON" -m pip install --no-index --find-links "$WHEELHOUSE" -r "$REQ"
elif [ -n "$PIP_INDEX_URL" ]; then
  echo "Installing dependencies with PIP_INDEX_URL=$PIP_INDEX_URL"
  "$PYTHON" -m pip install -i "$PIP_INDEX_URL" -r "$REQ"
else
  "$PYTHON" -m pip install -r "$REQ"
fi

"$PYTHON" scripts/check_env.py
