#!/usr/bin/env bash
set -euo pipefail
# 本地 quick output 检查脚本：只调用 starter_code/check_quick_outputs.py，
# 不训练、不预测，用于确认已有 quick 输出结构是否符合预期。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT/starter_code"
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi
# Jittor quick 检查也沿用单进程编译设置，避免本机环境抖动。
export DISABLE_MULTIPROCESSING=1
export cc_path="${cc_path:-/usr/bin/g++-12}"
export PYTHON="${PYTHON:-python3}"
"$PYTHON" check_quick_outputs.py
