#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
BASE_OUT="analysis/garad_v01_smoothing_adapter_confirm_20260518"
for seed in 20260519 20260520; do
  python scripts/train_garad_v0.py \
    --cpu \
    --base-mode paired-smooth \
    --train-limit 64 \
    --eval-limit 96 \
    --num-points 512 \
    --batch-size 2 \
    --base-k 8 \
    --base-alpha 0.04 \
    --max-step 0.020 \
    --lambda-delta 1.0 \
    --lambda-offset 0.0 \
    --lambda-cd 1.0 \
    --model garad \
    --steps 160 \
    --log-every 40 \
    --seed "$seed" \
    --out-dir "$BASE_OUT/k8_a0p04_cd1p0_garad_eval96_seed${seed}"
done
python - <<'PY'
import csv, json
from pathlib import Path
base=Path('analysis/garad_v01_smoothing_adapter_confirm_20260518')
rows=[]
for path in sorted(base.glob('k8_a0p04_cd1p0_garad_eval96*/summary.json')):
    data=json.loads(path.read_text()); e=data['eval']; a=data['args']
    rows.append({
        'run': path.parent.name,
        'seed': a['seed'],
        'cd_noisy': e['cd_noisy'],
        'cd_base': e['cd_base'],
        'cd_pred': e['cd_pred'],
        'cd_gain': e['cd_base'] - e['cd_pred'],
        'base_better_than_noisy_rate': e['base_better_than_noisy_rate'],
        'pred_better_than_base_rate': e['pred_better_than_base_rate'],
        'score_vs_base': e['score_vs_base'],
        'delta_l2_mean': e['delta_l2_mean'],
        'gate_mean': e['gate_mean'],
        'pass_gate': int(e['cd_pred'] < e['cd_base'] and e['pred_better_than_base_rate'] >= 0.60),
    })
out=base/'k8_a0p04_cd1p0_garad_eval96_repeats.csv'
with out.open('w', newline='') as f:
    w=csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
lines=['# k8_a0p04 cd1p0 GARA-D eval96 repeat seeds','', '| run | seed | cd_base | cd_pred | cd_gain | win_rate | pass |', '|---|---:|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['run']} | {r['seed']} | {r['cd_base']:.8g} | {r['cd_pred']:.8g} | {r['cd_gain']:.8g} | {r['pred_better_than_base_rate']:.3f} | {r['pass_gate']} |")
lines += ['', f"Summary CSV: `{out.name}`"]
(base/'k8_a0p04_cd1p0_garad_eval96_repeats.md').write_text('\n'.join(lines)+'\n')
print(out)
print('\n'.join(lines))
PY
