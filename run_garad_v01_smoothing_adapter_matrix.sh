#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BASE_OUT="analysis/garad_v01_smoothing_adapter_matrix_20260518"
mkdir -p "$BASE_OUT"

# Round-1 noisy-only smoothing base adapter matrix.
# Candidates chosen from smoothing base scan:
# - k8_a0p04: stable first train candidate with meaningful but safe improvement
# - k8_a0p08: mean-CD stronger candidate, lower win-rate
# - k12_a0p02: high-stability control from a different k
CANDIDATES=(
  "k8:0.04:k8_a0p04"
  "k8:0.08:k8_a0p08"
  "k12:0.02:k12_a0p02"
)
CD_WEIGHTS=(0.1 1.0)

COMMON_BASE=(
  --cpu
  --base-mode paired-smooth
  --train-limit 64
  --eval-limit 32
  --num-points 512
  --batch-size 2
  --max-step 0.020
  --lambda-delta 1.0
  --lambda-offset 0.0
  --steps 160
  --log-every 40
)

for spec in "${CANDIDATES[@]}"; do
  IFS=":" read -r k alpha tag <<< "$spec"
  cand_out="$BASE_OUT/$tag"
  mkdir -p "$cand_out"

  # One zero/identity baseline per smoothing config. lambda_cd is irrelevant here.
  python scripts/train_garad_v0.py \
    "${COMMON_BASE[@]}" \
    --base-k "${k#k}" \
    --base-alpha "$alpha" \
    --lambda-cd 1.0 \
    --model zero \
    --steps 1 \
    --out-dir "$cand_out/zero"

  for cdw in "${CD_WEIGHTS[@]}"; do
    cdtag="cd${cdw//./p}"
    python scripts/train_garad_v0.py \
      "${COMMON_BASE[@]}" \
      --base-k "${k#k}" \
      --base-alpha "$alpha" \
      --lambda-cd "$cdw" \
      --model residual_mlp \
      --out-dir "$cand_out/${cdtag}_residual_mlp"

    python scripts/train_garad_v0.py \
      "${COMMON_BASE[@]}" \
      --base-k "${k#k}" \
      --base-alpha "$alpha" \
      --lambda-cd "$cdw" \
      --model garad \
      --out-dir "$cand_out/${cdtag}_garad"
  done
done

python scripts/summarize_garad_adapter_matrix.py "$BASE_OUT"
