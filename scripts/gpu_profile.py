#!/usr/bin/env python3
"""Suggest a training profile from visible NVIDIA GPU names/VRAM.
This does not change configs; it only prints a safe recommendation.
"""
import re
import subprocess

try:
    out = subprocess.check_output([
        'nvidia-smi',
        '--query-gpu=name,memory.total',
        '--format=csv,noheader,nounits'
    ], text=True)
except Exception as e:
    print('Could not query GPU:', e)
    print('Recommended profile: configs/profiles/local_dev.yaml')
    raise SystemExit(0)

gpus = []
for line in out.strip().splitlines():
    if not line.strip():
        continue
    name, mem = [x.strip() for x in line.split(',', 1)]
    mem = int(re.findall(r'\d+', mem)[0])
    gpus.append((name, mem))

print('Visible GPUs:')
for i, (name, mem) in enumerate(gpus):
    print(f'  [{i}] {name}, {mem} MiB')

max_mem = max((m for _, m in gpus), default=0)
names = ' '.join(n.lower() for n, _ in gpus)
if 'a6000' in names or max_mem >= 40000:
    rec = 'configs/profiles/a6000.yaml'
elif '5060' in names or max_mem >= 12000:
    rec = 'configs/profiles/rtx5060ti.yaml'
else:
    rec = 'configs/profiles/local_dev.yaml'

print('Recommended profile:', rec)
