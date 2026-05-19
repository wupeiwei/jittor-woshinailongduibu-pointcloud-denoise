#!/usr/bin/env bash
# Run self-written denoise_baseline checkpoint prediction on A6000, then package and validate.
# Intended after full training has produced experiments/denoise_baseline/baseline.pkl.
# 中文边界：这是自写 denoise_baseline 的 A6000 全量推理链路；
# 与官方 VM fixed-stitch 链路分开，避免两个 baseline 的 artifact 混淆。
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

CONFIG="${CONFIG:-configs/denoise_baseline.yaml}"
PROFILE="${PROFILE:-configs/profiles/a6000.yaml}"
CKPT="${CKPT:-experiments/denoise_baseline/baseline.pkl}"
OUT_DIR="${OUT_DIR:-results/denoise_baseline_a6000_full}"
ZIP_PATH="${ZIP_PATH:-result_denoise_baseline_a6000_full.zip}"

# 所有关键路径先检查，避免 Jittor/CUDA 初始化后才发现输入缺失。
require_file() {
  local path="$1"
  if [ ! -f "$path" ]; then
    echo "MISSING_FILE: $path" >&2
    exit 2
  fi
}

require_dir() {
  local path="$1"
  if [ ! -d "$path" ]; then
    echo "MISSING_DIR: $path" >&2
    exit 2
  fi
}

require_file "$CONFIG"
require_file "$PROFILE"
require_file "$CKPT"
require_dir "dataset_test_noisy"
require_file "scripts/env.sh"
require_file "scripts/check_submission.py"
require_file "denoise_baseline.py"

if ! nvidia-smi; then
  # 在 import Jittor 之前确认 GPU 可见，减少 CUDA 编译失败后的排查成本。
  echo "GPU_NOT_READY: nvidia-smi failed. Stop before importing Jittor." >&2
  exit 10
fi

if [ -f scripts/activate_a6000_workspace.sh ]; then
  # shellcheck disable=SC1091
  source scripts/activate_a6000_workspace.sh
else
  export PYTHON=/workspace/freshman/env/conda/bin/python
  export HOME=/workspace/freshman/env/home
  export XDG_CACHE_HOME=/workspace/freshman/env/cache
  export TMPDIR=/workspace/freshman/env/tmp
  export PATH=/workspace/freshman/env/conda/bin:$PATH
  export cc_path=/workspace/freshman/env/conda/bin/x86_64-conda-linux-gnu-g++
  export CC=/workspace/freshman/env/conda/bin/x86_64-conda-linux-gnu-gcc
  export CXX=$cc_path
  export DISABLE_MULTIPROCESSING=1
  export PYTHONNOUSERSITE=1
  # shellcheck disable=SC1091
  source scripts/env.sh
fi

if [ -z "${PYTHON:-}" ]; then
  echo "PYTHON is not set after sourcing scripts/env.sh" >&2
  exit 2
fi

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "PYTHON=$PYTHON"
echo "CONFIG=$CONFIG"
echo "PROFILE=$PROFILE"
echo "CKPT=$CKPT"
echo "OUT_DIR=$OUT_DIR"
echo "ZIP_PATH=$ZIP_PATH"

echo "=== predict ==="
# 三步显式拆开：predict 写 out_dir，zip 打包，validate/check_submission 做结构和数值校验。
"$PYTHON" denoise_baseline.py \
  --config "$CONFIG" \
  --profile "$PROFILE" \
  --ckpt "$CKPT" \
  --out-dir "$OUT_DIR" \
  --zip "$ZIP_PATH" \
  --mode predict

echo "=== zip ==="
"$PYTHON" denoise_baseline.py \
  --config "$CONFIG" \
  --profile "$PROFILE" \
  --ckpt "$CKPT" \
  --out-dir "$OUT_DIR" \
  --zip "$ZIP_PATH" \
  --mode zip

echo "=== validate zip entries ==="
"$PYTHON" denoise_baseline.py \
  --config "$CONFIG" \
  --profile "$PROFILE" \
  --ckpt "$CKPT" \
  --out-dir "$OUT_DIR" \
  --zip "$ZIP_PATH" \
  --mode validate-zip

echo "=== submission check ==="
"$PYTHON" scripts/check_submission.py "$ZIP_PATH" \
  --config "$CONFIG" \
  --profile "$PROFILE" \
  --require-float32

echo "=== artifact summary ==="
ls -lh "$ZIP_PATH"
sha256sum "$ZIP_PATH"
find "$OUT_DIR" -path '*/denoised.npy' | wc -l | awk '{print "denoised_files=" $1}'
echo "DONE: $ZIP_PATH"
