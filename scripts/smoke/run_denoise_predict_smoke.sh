#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"
export PATH="$PWD/starter_code/.toolchain-gcc10:$PATH"
export CC=gcc
export CXX=g++
export DISABLE_MULTIPROCESSING=1
source starter_code/.venv/bin/activate
python denoise_baseline.py \
  --mode predict \
  --limit 1 \
  --feat-dim 64 \
  --hidden 64 \
  --k 8 \
  --ckpt experiments/denoise_baseline/smoke.pkl \
  --out-dir results/denoise_smoke
python denoise_baseline.py \
  --mode zip \
  --out-dir results/denoise_smoke \
  --zip result_denoise_smoke.zip
