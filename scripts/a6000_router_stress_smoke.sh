#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/activate_a6000_workspace.sh >/dev/null 2>&1 || source scripts/env.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DISABLE_MULTIPROCESSING=1
export cc_path="${cc_path:-${CXX:-g++}}"

LIMIT="${LIMIT:-8}"
THRESHOLD="${THRESHOLD:-0.016}"
PATCH_SIZE="${PATCH_SIZE:-8192}"
SOFT_WIDTH="${SOFT_WIDTH:-0.001}"
PROFILE="${PROFILE:-configs/profiles/a6000.yaml}"
SAFE_CONFIG="${SAFE_CONFIG:-configs/denoise_pwsenel_v2_adaptive_clip_piecewise.yaml}"
STRONG_CONFIG="${STRONG_CONFIG:-configs/denoise_noise_aware_move_gate.yaml}"
SAFE_CKPT="${SAFE_CKPT:-experiments/denoise_pwsenel_v2_adaptive_clip_piecewise/pwsenel_v2_adaptive_clip_piecewise.pkl}"
STRONG_CKPT="${STRONG_CKPT:-experiments/denoise_noise_aware_move_gate/noise_aware_move_gate.pkl}"
TEST_ROOT="${TEST_ROOT:-dataset_test_noisy}"

for mode in hard soft force-safe force-strong; do
  out="results/router_stress_${mode}_limit${LIMIT}_t${THRESHOLD}"
  zip="result_router_stress_${mode}_limit${LIMIT}_t${THRESHOLD}.zip"
  echo "=== $mode ==="
  "$PYTHON" scripts/unified_predict.py     --name "router_stress_${mode}_limit${LIMIT}_t${THRESHOLD}"     --safe-config "$SAFE_CONFIG"     --strong-config "$STRONG_CONFIG"     --profile "$PROFILE"     --safe-ckpt "$SAFE_CKPT"     --strong-ckpt "$STRONG_CKPT"     --test-root "$TEST_ROOT"     --out-dir "$out"     --zip "$zip"     --limit "$LIMIT"     --patch-size "$PATCH_SIZE"     --threshold "$THRESHOLD"     --router-mode "$mode"     --soft-width "$SOFT_WIDTH"     --no-zip
  echo
 done
