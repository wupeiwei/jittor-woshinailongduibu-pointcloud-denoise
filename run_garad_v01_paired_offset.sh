#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
BASE_OUT="analysis/garad_v01_paired_offset_20260518"
mkdir -p "$BASE_OUT"
COMMON=(--cpu --base-mode paired --train-limit 64 --eval-limit 32 --num-points 512 --batch-size 2 --base-alpha 0.08 --max-step 0.020 --lambda-cd 0.0 --lambda-delta 1.0 --lambda-offset 0.0 --steps 160 --log-every 40)
python scripts/train_garad_v0.py "${COMMON[@]}" --model zero --steps 1 --out-dir "$BASE_OUT/zero"
python scripts/train_garad_v0.py "${COMMON[@]}" --model residual_mlp --out-dir "$BASE_OUT/residual_mlp_delta_only"
python scripts/train_garad_v0.py "${COMMON[@]}" --model garad --out-dir "$BASE_OUT/garad_delta_only"
python - <<'PY'
import csv, json
from pathlib import Path
base=Path('analysis/garad_v01_paired_offset_20260518')
rows=[]
for path in sorted(base.glob('*/summary.json')):
    data=json.loads(path.read_text()); e=data['eval']; args=data['args']
    rows.append({
        'run': path.parent.name,
        'model': args['model'],
        'cd_noisy': e['cd_noisy'],
        'cd_base': e['cd_base'],
        'cd_pred': e['cd_pred'],
        'base_better_than_noisy_rate': e['base_better_than_noisy_rate'],
        'pred_better_than_base_rate': e['pred_better_than_base_rate'],
        'score_vs_base': e['score_vs_base'],
        'delta_l2_mean': e['delta_l2_mean'],
    })
out=base/'paired_summary.csv'
with out.open('w', newline='') as f:
    w=csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(out)
for r in rows:
    print(r)
PY
