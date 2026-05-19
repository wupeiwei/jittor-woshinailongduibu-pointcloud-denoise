#!/usr/bin/env bash
set -euo pipefail

# A6000 bounded runner for LIR-v1 T=2 fairness ablation.
# Usage:
#   bash scripts/a6000_lir_v1_t2_fast_ablation.sh smoke
#   bash scripts/a6000_lir_v1_t2_fast_ablation.sh baseline-smoke
#   bash scripts/a6000_lir_v1_t2_fast_ablation.sh lir-fast
#   bash scripts/a6000_lir_v1_t2_fast_ablation.sh baseline-fast
#
# This script intentionally does not run official predict/zip/submit.

MODE="${1:-smoke}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

case "$MODE" in
  smoke|baseline-smoke|lir-fast|baseline-fast) ;;
  *)
    echo "Unknown mode: $MODE" >&2
    echo "Expected one of: smoke baseline-smoke lir-fast baseline-fast" >&2
    exit 2
    ;;
esac

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found; stop before Jittor training." >&2
  exit 3
fi

if ! nvidia-smi >/tmp/a6000_lir_v1_nvidia_smi.txt 2>&1; then
  echo "ERROR: nvidia-smi failed; GPU/NVML unavailable. Stop and fix GPU first." >&2
  cat /tmp/a6000_lir_v1_nvidia_smi.txt >&2 || true
  exit 4
fi
cat /tmp/a6000_lir_v1_nvidia_smi.txt

if [[ ! -f scripts/activate_a6000_workspace.sh ]]; then
  echo "ERROR: scripts/activate_a6000_workspace.sh missing; sync scripts before running." >&2
  exit 5
fi
# shellcheck source=/dev/null
source scripts/activate_a6000_workspace.sh

mkdir -p experiments/lir_v1_t2_fast_ablation logs

echo "=== environment summary ==="
echo "PWD=$PWD"
echo "PYTHON=$(command -v python)"
echo "CC=${CC:-}"
echo "CXX=${CXX:-}"
echo "cc_path=${cc_path:-}"
python - <<'PY'
import sys
print('python_version=', sys.version.replace('\n', ' '))
try:
    import jittor as jt
    print('jittor=', getattr(jt, '__version__', 'unknown'))
except Exception as e:
    print('jittor_import_error=', repr(e))
    raise
PY

case "$MODE" in
  smoke)
    CONFIG=configs/denoise_lir_v1_t2_fast_ablation.yaml
    EXTRA=(--steps 2 --limit 4 --num-points 512 --batch-size 1 --cd-every 1 --cd-num-points 256 --save-every 2 --log-every 1)
    LOG=logs/lir_v1_t2_smoke.log
    ;;
  baseline-smoke)
    CONFIG=configs/denoise_noise_aware_move_gate_fast_ablation.yaml
    EXTRA=(--steps 2 --limit 4 --num-points 512 --batch-size 1 --cd-every 1 --cd-num-points 256 --save-every 2 --log-every 1)
    LOG=logs/noise_aware_fast_baseline_smoke.log
    ;;
  lir-fast)
    CONFIG=configs/denoise_lir_v1_t2_fast_ablation.yaml
    EXTRA=()
    LOG=logs/lir_v1_t2_fast_ablation.log
    ;;
  baseline-fast)
    CONFIG=configs/denoise_noise_aware_move_gate_fast_ablation.yaml
    EXTRA=()
    LOG=logs/noise_aware_fast_baseline_ablation.log
    ;;
esac

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config missing: $CONFIG" >&2
  exit 6
fi

echo "=== run ==="
echo "mode=$MODE"
echo "config=$CONFIG"
echo "log=$LOG"
python denoise_baseline.py --config "$CONFIG" --mode train "${EXTRA[@]}" 2>&1 | tee "$LOG"
