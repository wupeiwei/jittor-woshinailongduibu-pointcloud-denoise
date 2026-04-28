#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$PROJECT_ROOT"
export PATH="$PWD/starter_code/.toolchain-gcc10:$PATH"
export CC=gcc
export CXX=g++
export DISABLE_MULTIPROCESSING=1
source starter_code/.venv/bin/activate
python -m py_compile denoise_baseline.py
python denoise_baseline.py \
  --mode train \
  --steps 2 \
  --limit 2 \
  --num-points 256 \
  --batch-size 1 \
  --feat-dim 64 \
  --hidden 64 \
  --k 8 \
  --log-every 1 \
  --save-every 2 \
  --ckpt experiments/denoise_baseline/smoke.pkl
