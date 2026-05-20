#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
BASE_OUT="analysis/garad_v01_base_modes_20260518"
mkdir -p "$BASE_OUT"
COMMON=(--cpu --steps 80 --train-limit 64 --eval-limit 32 --num-points 512 --batch-size 2 --base-alpha 0.08 --max-step 0.012 --lambda-delta 1.0 --lambda-offset 0.01)
for mode in easy medium hard; do
  python scripts/train_garad_v0.py "${COMMON[@]}" --base-mode "$mode" --model zero --steps 1 --out-dir "$BASE_OUT/${mode}_zero"
  python scripts/train_garad_v0.py "${COMMON[@]}" --base-mode "$mode" --model garad --out-dir "$BASE_OUT/${mode}_garad_delta"
  python scripts/train_garad_v0.py "${COMMON[@]}" --base-mode "$mode" --model residual_mlp --out-dir "$BASE_OUT/${mode}_residual_mlp_delta"
done
python - <<'PY'
import json, csv
from pathlib import Path
base=Path('analysis/garad_v01_base_modes_20260518')
rows=[]
for path in sorted(base.glob('*/summary.json')):
    data=json.loads(path.read_text())
    e=data['eval']; args=data['args']
    rows.append({
        'run': path.parent.name,
        'base_mode': args['base_mode'],
        'model': args['model'],
        'cd_noisy': e['cd_noisy'],
        'cd_base': e['cd_base'],
        'cd_pred': e['cd_pred'],
        'base_better_than_noisy_rate': e['base_better_than_noisy_rate'],
        'pred_better_than_base_rate': e['pred_better_than_base_rate'],
        'score_vs_base': e['score_vs_base'],
        'delta_l2_mean': e['delta_l2_mean'],
    })
out=base/'matrix_summary.csv'
with out.open('w', newline='') as f:
    w=csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(out)
for r in rows:
    print(r)
PY
