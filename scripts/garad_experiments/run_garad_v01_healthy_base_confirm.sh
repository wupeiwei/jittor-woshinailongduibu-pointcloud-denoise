#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export CUDA_VISIBLE_DEVICES=""
export DISABLE_MULTIPROCESSING=1
export nvcc_path=""
ROOT="analysis/garad_v01_healthy_base_confirm_20260519"
COMMON=(
  --cpu
  --train-limit 32
  --eval-limit 48
  --steps 180
  --batch-size 2
  --num-points 512
  --log-every 45
)
BASE_A=(
  --base-name pwsenel_v2_clip_w0p6_smooth0p5
  --base-ckpt experiments/denoise_pwsenel_v2_clip/pwsenel_v2_clip.pkl
  --base-model-pwsenel-v2
  --base-model-pwsenel-v2-edge-lock 0.7
  --base-model-pwsenel-v2-gate-scale 0.5
  --base-model-residual-clip 0.01
  --stream-weight 0.6
  --smooth-mix 0.5
)
BASE_B=(
  --base-name hybrid_safe_strong_w1_smooth0p5
  --base-ckpt experiments/denoise_hybrid_safe_strong/hybrid_safe_strong.pkl
  --base-model-pwsenel-v2
  --base-model-pwsenel-v2-edge-lock 0.7
  --base-model-pwsenel-v2-gate-scale 0.5
  --base-model-move-gate
  --base-model-hybrid-safe-strong
  --base-model-hybrid-router-scale 1.0
  --base-model-adaptive-clip
  --base-model-adaptive-clip-min 0.006
  --base-model-adaptive-clip-mid 0.012
  --base-model-adaptive-clip-max 0.028
  --base-model-adaptive-clip-ref-low 0.022
  --base-model-adaptive-clip-ref-mid 0.028
  --base-model-adaptive-clip-ref-high 0.036
  --stream-weight 1.0
  --smooth-mix 0.5
)
run_one() {
  local base_name="$1"; shift
  local model_name="$1"; shift
  local seed="$1"; shift
  local out="$ROOT/${base_name}/${model_name}_seed${seed}"
  echo "=== RUN $out ==="
  python scripts/train_garad_fixed075_proxy.py "${COMMON[@]}" --out-dir "$out" --seed "$seed" "$@"
}
for seed in 20260519 20260520 20260521; do
  run_one baseA zero "$seed" "${BASE_A[@]}" --model zero --steps 1
  run_one baseA residual_mlp_cd0p1 "$seed" "${BASE_A[@]}" --model residual_mlp --lambda-cd 0.1 --lambda-delta 1.0
  run_one baseA garad_cd1p0 "$seed" "${BASE_A[@]}" --model garad --lambda-cd 1.0 --lambda-delta 1.0
  run_one baseB zero "$seed" "${BASE_B[@]}" --model zero --steps 1
  run_one baseB residual_mlp_cd0p1 "$seed" "${BASE_B[@]}" --model residual_mlp --lambda-cd 0.1 --lambda-delta 1.0
  run_one baseB garad_cd1p0 "$seed" "${BASE_B[@]}" --model garad --lambda-cd 1.0 --lambda-delta 1.0
done
python - <<'PY'
import json,csv
from pathlib import Path
root=Path('analysis/garad_v01_healthy_base_confirm_20260519')
rows=[]
for p in sorted(root.glob('*/*/summary.json')):
    d=json.loads(p.read_text()); e=d['eval']; a=d['args']
    rows.append({
        'base': p.parent.parent.name,
        'run': p.parent.name,
        'seed': a['seed'],
        'base_name': a.get('base_name',''),
        'model': a['model'],
        'lambda_cd': a['lambda_cd'],
        'cd_noisy': e['cd_noisy'],
        'cd_base': e['cd_base'],
        'cd_pred': e['cd_pred'],
        'cd_gain': e['cd_base']-e['cd_pred'],
        'base_rate': e['base_better_than_noisy_rate'],
        'win_rate': e['pred_better_than_base_rate'],
        'delta_l2_mean': e['delta_l2_mean'],
        'gate_mean': e['gate_mean'],
        'pass_gate': int(e['cd_pred'] < e['cd_base'] and e['pred_better_than_base_rate'] >= 0.60),
    })
with (root/'summary.csv').open('w', newline='') as f:
    w=csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
lines=['# GARA-D healthy-base confirm 20260519','', '| base | run | seed | model | cd_base | cd_pred | cd_gain | base_rate | win_rate | pass |', '|---|---|---:|---|---:|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['base']} | {r['run']} | {r['seed']} | {r['model']} | {r['cd_base']:.8g} | {r['cd_pred']:.8g} | {r['cd_gain']:.8g} | {r['base_rate']:.3f} | {r['win_rate']:.3f} | {r['pass_gate']} |")
lines += ['', 'Gate: `cd_pred < cd_base` and `pred_better_than_base_rate >= 0.60`.', 'A6000 remains blocked unless healthy-base adapter repeats are stable and GARA-D is competitive with residual MLP.']
(root/'diagnosis.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
PY
