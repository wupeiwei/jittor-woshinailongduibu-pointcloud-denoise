#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"
source scripts/env.sh

REQ="${REQ:-requirements.txt}"
OUT="${OUT:-wheelhouse}"
mkdir -p "$OUT"

"$PYTHON" -m pip --version >/dev/null
"$PYTHON" -m pip install --upgrade pip wheel
"$PYTHON" -m pip download -r "$REQ" -d "$OUT"

tar -czf "${OUT}.tar.gz" "$OUT"
echo "Created: $PROJECT_ROOT/${OUT}.tar.gz"
