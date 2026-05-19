#!/usr/bin/env bash
set -euo pipefail

# Dry-run validation for the official fixed-stitch submission chain.
# This does NOT run full prediction.
# It only verifies that an existing fixed-stitch output tree can be
# repackaged into a formal zip and passes submission checks.
#
# Usage:
#   bash scripts/check_official_chain_dry_run.sh [output_root] [zip_path]
#
# Defaults assume the current best A6000-style tree exists under:
#   starter_code/tmp_predict_fixed_stitch
# and write the temp zip under /tmp.
#
# 中文边界：dry-run 只验证“已有输出树 -> 正式 zip -> submission check”链路，
# 不重新跑 VM 推理，适合提交前快速确认打包逻辑没有漂移。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

OUTPUT_ROOT="${1:-starter_code/tmp_predict_fixed_stitch}"
ZIP_PATH="${2:-/tmp/result_official_vm_fixed_stitch_dryrun.zip}"

if [ ! -d "$OUTPUT_ROOT" ]; then
  echo "MISSING_OUTPUT_ROOT: $OUTPUT_ROOT" >&2
  exit 20
fi

if [ -f scripts/activate_a6000_workspace.sh ]; then
  # shellcheck disable=SC1091
  source scripts/activate_a6000_workspace.sh
else
  # dry-run 可以只依赖 Python/Numpy 工具，不强制激活完整 Jittor 环境。
  export PYTHON="${PYTHON:-python3}"
fi

export DISABLE_MULTIPROCESSING=1
export cc_path="${cc_path:-/usr/bin/g++-12}"

echo "[1/3] Package existing fixed-stitch outputs"
# 从现有输出目录重新截取 shapenet/.../denoised.npy，防止 writer 嵌套路径污染 zip。
"${PYTHON:-python3}" scripts/package_official_vm_outputs.py \
  --output-root "$OUTPUT_ROOT" \
  --zip "$ZIP_PATH" \
  --expected-count 200 \
  --require-float32

echo "[2/3] Validate formal zip"
"${PYTHON:-python3}" scripts/check_submission.py "$ZIP_PATH" \
  --test-root dataset_test_noisy \
  --expected-count 200 \
  --require-float32

echo "[3/3] Summarize zip hash"
sha256sum "$ZIP_PATH"

echo "OFFICIAL_CHAIN_DRY_RUN_OK"
