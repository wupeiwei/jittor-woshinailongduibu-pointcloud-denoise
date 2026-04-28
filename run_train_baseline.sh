#!/usr/bin/env bash
set -euo pipefail
cd /home/sallen/jittor-pointcloud-denoise

export PATH="$PWD/starter_code/.toolchain-gcc10:$PATH"
export CC=gcc
export CXX=g++
export DISABLE_MULTIPROCESSING=1

source starter_code/.venv/bin/activate

python denoise_baseline.py \
  --mode train \
  --steps 5000 \
  --limit 2000 \
  --num-points 2048 \
  --batch-size 2 \
  --feat-dim 256 \
  --hidden 256 \
  --k 16 \
  --lr 1e-4 \
  --log-every 20 \
  --save-every 500 \
  --ckpt experiments/denoise_baseline/baseline.pkl
