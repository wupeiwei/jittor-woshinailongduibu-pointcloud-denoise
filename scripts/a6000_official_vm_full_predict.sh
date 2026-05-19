#!/usr/bin/env bash
# Run official starter_code VM full prediction on A6000 and package it into formal submission zip.
# Usage on A6000:
#   cd /workspace/freshman/jittor-pointcloud-denoise
#   bash scripts/a6000_official_vm_full_predict.sh
#
# Safety rules: no sudo, no apt, no system pollution. Stops if nvidia-smi fails.
# Important: official configs/task/predict_vm.yaml uses transform: vm, which expects mesh vertices.
# For noisy .npy test prediction, this script writes a fixed task config with transform: predict.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/freshman/jittor-pointcloud-denoise}"
cd "$PROJECT_ROOT"

echo "[1/9] GPU health check"
if ! nvidia-smi; then
  echo "GPU_NOT_READY: nvidia-smi failed. Stop before importing Jittor." >&2
  exit 10
fi

echo "[2/9] Check existing python/jittor jobs"
pgrep -af "python|jittor|train|predict|run.py" || true
cat <<'MSG'
If the process list above shows another active Jittor training/prediction job, press Ctrl+C now.
MSG
sleep 5

echo "[3/9] Activate A6000 workspace env"
if [ -f scripts/activate_a6000_workspace.sh ]; then
  # shellcheck disable=SC1091
  source scripts/activate_a6000_workspace.sh
else
  export PYTHON=/workspace/freshman/env/conda/bin/python
  export HOME=/workspace/freshman/env/home
  export PYTHONNOUSERSITE=1
  # shellcheck disable=SC1091
  source scripts/env.sh
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DISABLE_MULTIPROCESSING=1
export cc_path="${cc_path:-${CXX:-g++}}"

echo "PYTHON=${PYTHON:-python}"
"${PYTHON:-python}" --version

echo "[4/9] Check required files"
test -d starter_code
test -d starter_code/dataset_test_noisy
test -f starter_code/configs/transform/predict.yaml
test -f starter_code/experiments/vm/checkpoint_99.pkl
test -f scripts/package_official_vm_outputs.py
test -f scripts/check_submission.py

echo "[5/9] Write fixed full predict task config"
cat > starter_code/configs/task/predict_vm_full_fixed.yaml <<'YAML'
mode: predict
debug: False
load_ckpt: experiments/vm/checkpoint_99.pkl

components:
  data: predict
  transform: predict
  system: vm
  model: vm

writer:
  __target__: vm
  save_dir: tmp_predict
  save_name: denoised
YAML

echo "[6/9] Clean old official VM full output/zip"
rm -rf starter_code/tmp_predict result_official_vm_full.zip

echo "[7/9] Run official VM full prediction"
(
  cd starter_code
  "${PYTHON:-python}" run.py --task configs/task/predict_vm_full_fixed.yaml
)

echo "[8/9] Package formal zip"
"${PYTHON:-python}" scripts/package_official_vm_outputs.py \
  --output-root starter_code/tmp_predict \
  --zip result_official_vm_full.zip \
  --expected-count 200 \
  --require-float32

echo "[9/9] Validate zip"
"${PYTHON:-python}" scripts/check_submission.py result_official_vm_full.zip \
  --test-root dataset_test_noisy \
  --expected-count 200 \
  --require-float32

echo "OFFICIAL_VM_FULL_OK"
sha256sum result_official_vm_full.zip
