#!/usr/bin/env bash
# Source this file before running training/prediction.
# It is intentionally conservative for Jittor + CUDA compatibility across local RTX 4050 and remote A6000 machines.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

# Prefer local gcc/g++-10 wrapper when present. On an A6000 server, install gcc-10/g++-10 if Jittor rejects the default compiler.
if [ -d "$PROJECT_ROOT/starter_code/.toolchain-gcc10" ]; then
  export PATH="$PROJECT_ROOT/starter_code/.toolchain-gcc10:$PATH"
fi

export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"
export DISABLE_MULTIPROCESSING="${DISABLE_MULTIPROCESSING:-1}"
# Keep Jittor cache writes inside an explicit repository-local scratch root by
# default. Jittor resolves cache paths under "$JITTOR_HOME/.cache/jittor"; users
# can override this for shared compiled caches on their own machines.
export JITTOR_HOME="${JITTOR_HOME:-$PROJECT_ROOT/.jittor_home}"

# Optional venv. On A6000/5060Ti servers, create the same path or set VENV_PATH=/path/to/venv.
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/starter_code/.venv}"
if [ -f "$VENV_PATH/bin/activate" ]; then
  source "$VENV_PATH/bin/activate"
fi

# Python executable compatibility: many servers only provide `python3`, not `python`.
# Users may override with: PYTHON=/path/to/python bash scripts/train.sh ...
if [ -z "${PYTHON:-}" ]; then
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
