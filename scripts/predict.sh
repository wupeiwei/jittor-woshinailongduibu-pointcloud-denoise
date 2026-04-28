#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/denoise_baseline.yaml}"
PROFILE="${2:-}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"
source scripts/env.sh
EXTRA_ARGS=()
if [ -n "$PROFILE" ]; then EXTRA_ARGS+=(--profile "$PROFILE"); fi
"$PYTHON" denoise_baseline.py --config "$CONFIG" "${EXTRA_ARGS[@]}" --mode predict
"$PYTHON" denoise_baseline.py --config "$CONFIG" "${EXTRA_ARGS[@]}" --mode zip
"$PYTHON" denoise_baseline.py --config "$CONFIG" "${EXTRA_ARGS[@]}" --mode validate-zip
