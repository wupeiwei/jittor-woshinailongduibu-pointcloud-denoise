#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/denoise_baseline.yaml}"
PROFILE="${2:-}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"
source scripts/env.sh

EXP_NAME=$("$PYTHON" - <<PY
import yaml
cfg=yaml.safe_load(open('$CONFIG'))
print(cfg.get('experiment',{}).get('name','experiment'))
PY
)
RUN_ID="${EXP_NAME}_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="experiments/runs/$RUN_ID"
mkdir -p "$RUN_DIR"
cp "$CONFIG" "$RUN_DIR/config.yaml"
if [ -n "$PROFILE" ]; then cp "$PROFILE" "$RUN_DIR/profile.yaml"; fi
"$PYTHON" scripts/check_env.py > "$RUN_DIR/env.txt" 2>&1 || { cat "$RUN_DIR/env.txt"; exit 1; }
DATA_ARGS=(--config "$CONFIG" --limit 3)
if [ -n "$PROFILE" ]; then DATA_ARGS+=(--profile "$PROFILE"); fi
"$PYTHON" scripts/check_data.py "${DATA_ARGS[@]}" > "$RUN_DIR/data.txt" 2>&1 || { cat "$RUN_DIR/data.txt"; exit 1; }
EXTRA_ARGS=()
if [ -n "$PROFILE" ]; then EXTRA_ARGS+=(--profile "$PROFILE"); fi
{
  echo "config: $CONFIG"
  echo "profile: ${PROFILE:-none}"
  echo "run_dir: $RUN_DIR"
  echo "python: $PYTHON"
  echo "command: \"$PYTHON\" denoise_baseline.py --config $CONFIG ${EXTRA_ARGS[*]:-} --mode train"
  echo "date: $(date -Is)"
  git rev-parse --is-inside-work-tree >/dev/null 2>&1 && git status --short || true
} > "$RUN_DIR/meta.txt"

"$PYTHON" denoise_baseline.py --config "$CONFIG" "${EXTRA_ARGS[@]}" --mode train 2>&1 | tee "$RUN_DIR/train.log"
cp "$CONFIG" "$RUN_DIR/config.final.yaml"
echo "$RUN_DIR"
