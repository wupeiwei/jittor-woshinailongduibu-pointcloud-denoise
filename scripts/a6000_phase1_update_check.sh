#!/usr/bin/env bash
# A6000 Phase-1 update/predict checklist for the formal denoising competition.
# Run on the A6000 machine only, under /workspace/freshman.
# Safety rules: no sudo, no apt, no system pollution, no concurrent Jittor jobs.

set -euo pipefail

echo "[1/7] GPU health check"
nvidia-smi

echo "[2/7] Enter project"
cd "${PROJECT_ROOT:-/workspace/freshman/jittor-pointcloud-denoise}"

if command -v git >/dev/null 2>&1; then
  echo "[3/7] Git status before update"
  git status --short || true
else
  echo "git not found; update via tar/scp from local workspace instead" >&2
fi

echo "[4/7] Activate project/A6000 environment"
if [ -f scripts/activate_a6000_workspace.sh ]; then
  # shellcheck disable=SC1091
  source scripts/activate_a6000_workspace.sh
elif [ -f scripts/env.sh ]; then
  export PYTHON="${PYTHON:-/workspace/freshman/env/conda/bin/python}"
  export HOME="${HOME:-/workspace/freshman/env/home}"
  export PYTHONNOUSERSITE=1
  # shellcheck disable=SC1091
  source scripts/env.sh
else
  export PYTHON="${PYTHON:-/workspace/freshman/env/conda/bin/python}"
  export HOME="${HOME:-/workspace/freshman/env/home}"
  export PYTHONNOUSERSITE=1
fi

echo "PYTHON=${PYTHON:-python}"
"${PYTHON:-python}" --version

echo "[5/7] Static checks only"
"${PYTHON:-python}" -m py_compile \
  scripts/candidate_registry.py \
  scripts/unified_predict.py \
  scripts/run_official_eval.py \
  scripts/check_submission.py
"${PYTHON:-python}" scripts/unified_predict.py --help >"${TMPDIR:-/tmp}/unified_predict_help.txt"
"${PYTHON:-python}" scripts/candidate_registry.py --rebuild-only

echo "[6/7] Required checkpoint/data presence"
test -d dataset_test_noisy
ls -lh \
  experiments/denoise_pwsenel_v2_adaptive_clip_piecewise/pwsenel_v2_adaptive_clip_piecewise.pkl \
  experiments/denoise_noise_aware_move_gate/noise_aware_move_gate.pkl

echo "[7/7] Optional router_t0165 prediction command (do not run if another Jittor job is active)"
echo 'nvidia-smi'
echo 'pgrep -af "python|jittor|train|predict" || true'
echo 'source scripts/activate_a6000_workspace.sh'
echo '$PYTHON scripts/unified_predict.py --name router_t0165 --threshold 0.0165 --patch-size 8192 --out-dir results/denoise_router_t0165 --zip result_denoise_router_t0165.zip --append-registry'
echo '$PYTHON scripts/check_submission.py result_denoise_router_t0165.zip --test-root dataset_test_noisy --require-float32'
