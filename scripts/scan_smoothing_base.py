#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis/garad_v01_smoothing_base_scan_20260518'
OUT.mkdir(parents=True, exist_ok=True)

base_ks = [8, 12, 16]
base_alphas = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20]
rows = []

for k in base_ks:
    for alpha in base_alphas:
        tag = f'k{k}_a{str(alpha).replace(".", "p")}'
        run_dir = OUT / tag
        cmd = [
            sys.executable, 'scripts/train_garad_v0.py',
            '--cpu',
            '--base-mode', 'paired-smooth',
            '--train-limit', '8',
            '--eval-limit', '96',
            '--num-points', '512',
            '--batch-size', '2',
            '--base-k', str(k),
            '--base-alpha', str(alpha),
            '--max-step', '0.020',
            '--lambda-cd', '0.0',
            '--lambda-delta', '0.0',
            '--lambda-offset', '0.0',
            '--steps', '1',
            '--log-every', '1',
            '--model', 'zero',
            '--out-dir', str(run_dir),
        ]
        print('RUN', tag, flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
        data = json.loads((run_dir / 'summary.json').read_text())
        e = data['eval']
        row = {
            'base_k': k,
            'base_alpha': alpha,
            'cd_noisy': e['cd_noisy'],
            'cd_base': e['cd_base'],
            'base_better_than_noisy_rate': e['base_better_than_noisy_rate'],
            'score_vs_noisy': 100.0 * (1.0 - e['cd_base'] / max(e['cd_noisy'], 1e-12)),
            'run': tag,
        }
        rows.append(row)
        print(row, flush=True)

rows.sort(key=lambda r: (-(r['base_better_than_noisy_rate']), r['cd_base']))
out_csv = OUT / 'base_scan_summary.csv'
with out_csv.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

best = rows[:10]
md = OUT / 'diagnosis.md'
with md.open('w') as f:
    f.write('# GARA-D v0.1 noisy-only smoothing base scan\n\n')
    f.write('Zero-adapter/base-only scan for `paired-smooth` noisy-only smoothing base. No adapter training.\n\n')
    f.write('## Top candidates\n\n')
    f.write('| base_k | base_alpha | cd_noisy | cd_base | base>noisy rate | score_vs_noisy | run |\n')
    f.write('|---:|---:|---:|---:|---:|---:|---|\n')
    for r in best:
        f.write(f"| {r['base_k']} | {r['base_alpha']:.3f} | {r['cd_noisy']:.10f} | {r['cd_base']:.10f} | {r['base_better_than_noisy_rate']:.4f} | {r['score_vs_noisy']:.3f} | {r['run']} |\n")
    pass_rows = [r for r in rows if r['base_better_than_noisy_rate'] >= 0.80 and r['cd_base'] < r['cd_noisy']]
    f.write('\n## Gate\n\n')
    if pass_rows:
        f.write(f'Pass candidates: {len(pass_rows)}. Recommended first train candidate: `{pass_rows[0]["run"]}`.\n')
    else:
        f.write('No candidate reached `base_better_than_noisy_rate >= 0.80` with mean `cd_base < cd_noisy`.\n')

print(out_csv)
print(md)
print('TOP10')
for r in best:
    print(r)
