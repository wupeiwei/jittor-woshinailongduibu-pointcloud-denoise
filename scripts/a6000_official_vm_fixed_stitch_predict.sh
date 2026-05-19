#!/usr/bin/env bash
# Run official starter_code VM prediction with the fixed patch-stitching logic.
# Usage on A6000:
#   cd /workspace/freshman/jittor-pointcloud-denoise
#   bash scripts/a6000_official_vm_fixed_stitch_predict.sh
#
# Safety rules: no sudo, no apt, no system pollution. Stops if nvidia-smi fails.
# This variant uses the same official VM checkpoint/config path as the repaired
# baseline, but with starter_code/src/model/vm.py coverage repair so every input
# point has exactly one output point.
# 中文边界：这是官方 VM + fixed-stitch 补丁的正式预测链路；不会训练模型，
# 只负责 GPU 环境检查、运行 starter_code 推理、打包和本地提交校验。

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
# 给操作者 5 秒人工确认，避免在共享 A6000 上和已有 Jittor 任务抢显存。
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
require_file() {
  local path="$1"
  if [ ! -f "$path" ]; then
    echo "MISSING_FILE: $path" >&2
    exit 20
  fi
}
require_dir() {
  local path="$1"
  if [ ! -d "$path" ]; then
    echo "MISSING_DIR: $path" >&2
    exit 21
  fi
}
require_dir starter_code
require_dir starter_code/dataset_test_noisy
require_file starter_code/configs/transform/predict.yaml
require_file starter_code/experiments/vm/checkpoint_99.pkl
require_file starter_code/src/model/vm.py
require_file scripts/package_official_vm_outputs.py
require_file scripts/check_submission.py

echo "[5/9] Write fixed full predict task config"
# 只写预测 task 配置，不改模型配置；VM 补丁来自 starter_code/src/model/vm.py。
cat > starter_code/configs/task/predict_vm_fixed_stitch.yaml <<'YAML'
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
  save_dir: tmp_predict_fixed_stitch
  save_name: denoised
YAML

echo "[6/9] Clean old fixed-stitch output/zip"
# 清理的是本脚本专用输出，避免旧 denoised.npy 混入新 zip。
rm -rf starter_code/tmp_predict_fixed_stitch result_official_vm_fixed_stitch.zip

echo "[7/9] Run official VM fixed-stitch full prediction"
(
  cd starter_code
  "${PYTHON:-python}" run.py --task configs/task/predict_vm_fixed_stitch.yaml
)

echo "[8/9] Package formal zip"
"${PYTHON:-python}" scripts/package_official_vm_outputs.py \
  --output-root starter_code/tmp_predict_fixed_stitch \
  --zip result_official_vm_fixed_stitch.zip \
  --expected-count 200 \
  --require-float32

echo "[9/9] Validate zip"
"${PYTHON:-python}" scripts/check_submission.py result_official_vm_fixed_stitch.zip \
  --test-root dataset_test_noisy \
  --expected-count 200 \
  --require-float32

echo "OFFICIAL_VM_FIXED_STITCH_OK"
sha256sum result_official_vm_fixed_stitch.zip
