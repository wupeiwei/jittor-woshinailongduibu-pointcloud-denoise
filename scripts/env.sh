#!/usr/bin/env bash
# Source this file before running training/prediction.
# It is intentionally conservative for Jittor + CUDA compatibility across local RTX 4050 and remote A6000 machines.

set -euo pipefail

# PROJECT_ROOT 允许外部覆盖；默认按 env.sh 所在目录反推仓库根目录。
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

# Prefer local gcc/g++-10 wrapper when present. On an A6000 server, install gcc-10/g++-10 if Jittor rejects the default compiler.
if [ -d "$PROJECT_ROOT/starter_code/.toolchain-gcc10" ]; then
  export PATH="$PROJECT_ROOT/starter_code/.toolchain-gcc10:$PATH"
fi

export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"
# Jittor 多进程编译在不少服务器环境里更容易触发不稳定问题，默认关闭。
export DISABLE_MULTIPROCESSING="${DISABLE_MULTIPROCESSING:-1}"
# Keep Jittor cache writes inside an explicit repository-local scratch root by
# default. Jittor resolves cache paths under "$JITTOR_HOME/.cache/jittor"; users
# can override this for shared compiled caches on their own machines.
# 默认使用独立的 .jittor_home，避免把 Jittor cache/临时文件散落到仓库根目录或用户 HOME。
export JITTOR_HOME="${JITTOR_HOME:-$PROJECT_ROOT/.jittor_home}"

# Optional venv. On A6000/5060Ti servers, create the same path or set VENV_PATH=/path/to/venv.
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/starter_code/.venv}"
if [ -f "$VENV_PATH/bin/activate" ]; then
  # 激活虚拟环境只改变当前 shell；调用方需要 `source scripts/env.sh` 才能继承。
  source "$VENV_PATH/bin/activate"
fi

# Python executable compatibility: many servers only provide `python3`, not `python`.
# Users may override with: PYTHON=/path/to/python bash scripts/train.sh ...
if [ -z "${PYTHON:-}" ]; then
  # 统一导出 PYTHON，后续 shell wrapper 不必猜 python/python3 哪个存在。
  if command -v python3 >/dev/null 2>&1; then
    export PYTHON="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    export PYTHON="$(command -v python)"
  else
    echo "ERROR: neither python3 nor python was found in PATH" >&2
    return 1 2>/dev/null || exit 1
  fi
else
  export PYTHON
fi
