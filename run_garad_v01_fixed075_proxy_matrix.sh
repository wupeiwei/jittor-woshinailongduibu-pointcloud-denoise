#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export CUDA_VISIBLE_DEVICES=""
export DISABLE_MULTIPROCESSING=1
export nvcc_path=""
ROOT="analysis/garad_v01_fixed075_proxy_matrix_20260519"
COMMON=(
  --cpu
  --train-limit 24
  --eval-limit 24
  --steps 160
  --batch-size 2
  --num-points 512
  --log-every 40
  --base-ckpt experiments/denoise_baseline/baseline.pkl
)
run_one() {
  local name="$1"; shift
  echo "=== RUN $name ==="
  python scripts/train_garad_fixed075_proxy.py "${COMMON[@]}" --out-dir "$ROOT/$name" "$@"
}
run_one zero --model zero --steps 1
run_one cd1p0_residual_mlp --model residual_mlp --lambda-cd 1.0 --lambda-delta 1.0
run_one cd1p0_garad --model garad --lambda-cd 1.0 --lambda-delta 1.0
run_one cd0p1_residual_mlp --model residual_mlp --lambda-cd 0.1 --lambda-delta 1.0
run_one cd0p1_garad --model garad --lambda-cd 0.1 --lambda-delta 1.0
python - <<'PY'
import json, csv
from pathlib import Path
root=Path('analysis/garad_v01_fixed075_proxy_matrix_20260519')
rows=[]
for p in sorted(root.glob('*/summary.json')):
    d=json.loads(p.read_text()); e=d['eval']; a=d['args']
    rows.append({
        'run': p.parent.name,
        'model': a['model'],
        'lambda_cd': a['lambda_cd'],
        'lambda_delta': a['lambda_delta'],
        'cd_noisy': e['cd_noisy'],
        'cd_base': e['cd_base'],
        'cd_pred': e['cd_pred'],
        'base_better_than_noisy_rate': e['base_better_than_noisy_rate'],
        'pred_better_than_base_rate': e['pred_better_than_base_rate'],
        'score_vs_base': e['score_vs_base'],
        'delta_l2_mean': e['delta_l2_mean'],
        'gate_mean': e['gate_mean'],
        'pass_gate': int(e['cd_pred'] < e['cd_base'] and e['pred_better_than_base_rate'] >= 0.60),
    })
with (root/'summary.csv').open('w', newline='') as f:
    w=csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
lines=['# GARA-D fixed075 proxy matrix 20260519','', '| run | model | lambda_cd | cd_base | cd_pred | win_rate | base_rate | pass |', '|---|---|---:|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['run']} | {r['model']} | {r['lambda_cd']} | {r['cd_base']:.8g} | {r['cd_pred']:.8g} | {r['pred_better_than_base_rate']:.3f} | {r['base_better_than_noisy_rate']:.3f} | {r['pass_gate']} |")
lines += ['', 'Gate: `cd_pred < cd_base` and `pred_better_than_base_rate >= 0.60`.', '', 'Decision: A6000 remains blocked unless GARA-D passes and repeat seeds confirm stability.']
(root/'diagnosis.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
PY
