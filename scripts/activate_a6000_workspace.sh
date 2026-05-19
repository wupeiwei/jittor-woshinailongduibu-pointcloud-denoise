#!/usr/bin/env bash
# Workspace-local runtime for the A6000 server.
# Source this file from the project root:
#   source scripts/activate_a6000_workspace.sh
#
# Everything is intentionally kept under /workspace/freshman to avoid
# polluting system Python, /home/.local, or shared server locations.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "This script must be sourced, not executed:" >&2
  echo "  source scripts/activate_a6000_workspace.sh" >&2
  exit 2
fi

export FRESHMAN_ROOT="${FRESHMAN_ROOT:-/workspace/freshman}"
export PROJECT_ROOT="${PROJECT_ROOT:-$FRESHMAN_ROOT/jittor-pointcloud-denoise}"
export ENV_ROOT="${ENV_ROOT:-$FRESHMAN_ROOT/env}"
export CONDA_PREFIX="$ENV_ROOT/conda"

export HOME="$ENV_ROOT/home"
export XDG_CACHE_HOME="$ENV_ROOT/cache"
export TMPDIR="$ENV_ROOT/tmp"
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$TMPDIR" "$ENV_ROOT/logs"

export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHON="$CONDA_PREFIX/bin/python"
export cc_path="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$cc_path"
export DISABLE_MULTIPROCESSING=1
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# Do not set CUDA_VISIBLE_DEVICES here. Choose explicitly before training if needed.
# Example:
#   export CUDA_VISIBLE_DEVICES=0

printf "A6000 workspace env activated\n"
printf "  PROJECT_ROOT=%s\n" "$PROJECT_ROOT"
printf "  PYTHON=%s\n" "$PYTHON"
printf "  CXX=%s\n" "$CXX"
printf "  HOME=%s\n" "$HOME"
printf "  XDG_CACHE_HOME=%s\n" "$XDG_CACHE_HOME"
printf "  TMPDIR=%s\n" "$TMPDIR"
