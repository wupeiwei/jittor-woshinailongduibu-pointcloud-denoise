#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"
source scripts/env.sh

OUT="${1:-experiments/env-freeze-$(date +%Y%m%d_%H%M%S).txt}"
mkdir -p "$(dirname "$OUT")"
{
  echo "date: $(date -Is)"
  echo "python: $PYTHON"
  "$PYTHON" --version
  echo
  echo "pip freeze:"
  "$PYTHON" -m pip freeze
  echo
  echo "nvidia-smi:"
  nvidia-smi || true
} > "$OUT"
echo "$OUT"
