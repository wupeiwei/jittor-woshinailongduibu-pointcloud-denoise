# ARCHIVE - ST-AAS / PW-SENEL Execution Checklist

Historical status: this document records the ST-AAS/PW-SENEL innovation plan
from an earlier phase. It is not the current formal submission boundary. For
current status, read `docs/repo_boundary_audit_20260519.md` and
`experiments/candidate_registry.md`.

Historical source plan omitted.

## Execution rule
Strictly follow the plan order. Do not jump to ST-AAS v1 or paper-stage work before Stage 1 is implemented, smoke-tested, and ablated against baseline.

## Stage 0: Baseline stability
- [x] Keep reproducible baseline config: `configs/denoise_baseline.yaml`
- [x] Keep GPU profiles, including A6000: `configs/profiles/a6000.yaml`
- [x] Collect complete training logs under `experiments/*/*.train.csv`
- [ ] Summarize baseline/A6000 evidence into a comparable run table

## Stage 1: ST-AAS v0
Plan requirements:
- Fixed KNN from existing implementation
- Local density / scale estimate
- Density-adaptive `tau_i`
- Structure tensor eigenvalue descriptors
- `edge_conf` smoothing suppression
- Switchable module, not replacing baseline

Implementation tasks:
- [x] Switch exists: `model.staas: true/false`, CLI `--staas`
- [x] Config exists: `configs/denoise_staas_v0.yaml`
- [ ] Code audit: verify v0 matches planned formula and does not mutate baseline path when disabled
- [ ] Run ST-AAS v0 smoke test
- [ ] Run small local/dev ablation
- [ ] Run A6000 Stage-1 ablation only after smoke passes

## Stage 2: ST-AAS v1
Blocked until Stage 1 results are reviewed.
- [ ] Ellipsoidal softmax metric
- [ ] Tensor-guided anisotropic distance
- [ ] Complete ablation

## Stage 3: Paper version
Blocked until Stage 1/2 results are strong enough.
- [ ] Runtime / memory benchmark
- [ ] Large-scale chunked inference evaluation
- [ ] Visualization and ablation tables

## Immediate next command sequence
1. Audit ST-AAS v0 code path.
2. Run smoke:
   `PYTHON=.venv/bin/python bash scripts/train.sh configs/denoise_staas_v0.yaml configs/profiles/local_dev.yaml --steps 3 --limit 4 --num-points 128 --batch-size 1 --feat-dim 64 --hidden 64 --k 8 --log-every 1 --save-every 3`
3. If smoke passes, run local/dev ablation.
4. If local/dev passes, run A6000 Stage-1 ablation with `configs/denoise_staas_v0.yaml configs/profiles/a6000.yaml`.
