#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_ROOT:-/workspace/freshman/jittor-pointcloud-denoise}"
source scripts/activate_a6000_workspace.sh

log_dir="/workspace/freshman/env/logs"
mkdir -p "$log_dir"
log_file="$log_dir/a6000_smoke_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$log_file") 2>&1

echo "# A6000 smoke check"
date
hostname
whoami
pwd

echo
echo "## Python / compiler"
"$PYTHON" --version
"$PYTHON" -m pip --version
"$cc_path" --version | head -1

echo
echo "## CPU-safe Python modules"
"$PYTHON" - <<'PY'
import importlib
for name in ["numpy", "yaml", "cupy"]:
    mod = importlib.import_module(name)
    print(f"{name}: {getattr(mod, '__version__', 'unknown')}")
PY

echo
echo "## GPU visibility"
if ! nvidia-smi; then
  echo "GPU_NOT_READY: nvidia-smi failed. Stop before importing Jittor CUDA/training."
  echo "log: $log_file"
  exit 10
fi

echo
echo "## Project env check, including Jittor"
"$PYTHON" scripts/check_env.py

echo
echo "## Data check"
"$PYTHON" scripts/check_data.py --config configs/denoise_baseline.yaml --profile configs/profiles/a6000.yaml --limit 5

echo
echo "SMOKE_OK"
echo "log: $log_file"
