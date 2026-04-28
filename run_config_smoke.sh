#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$PROJECT_ROOT"
source scripts/env.sh
python -m py_compile denoise_baseline.py scripts/check_env.py
python denoise_baseline.py \
  --config configs/denoise_baseline.yaml \
  --mode train \
  --steps 1 \
  --limit 1 \
  --num-points 128 \
  --batch-size 1 \
  --feat-dim 32 \
  --hidden 32 \
  --k 4 \
  --ckpt experiments/denoise_baseline/config_smoke.pkl \
  --log-every 1 \
  --save-every 1
