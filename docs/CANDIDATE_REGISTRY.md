# Candidate Registry and Unified Pipeline Notes

This document is a lightweight operator guide for Phase 1. It keeps candidate tracking, official evaluation, and A/B-isomorphic inference in one place.

Current best official reference in the registry:

- `blend_best075_lir025_20260517`
- official score `53.32`
- CD `40.65`
- P2S `65.99`

## Files

- `scripts/unified_predict.py` — Phase-1 `UnifiedDenoisePipeline` entry point. It currently implements noisy-only hard routing through `NoisyConditionedDenoiser`; future soft gate / LIR branches should be added behind the same interface.
- `scripts/candidate_registry.py` — automatic Candidate Registry writer.
- `scripts/run_official_eval.py` — wrapper for `starter_code/evaluate.py`; captures CD/P2S/final score into JSON for the registry.
- `experiments/candidates.jsonl` — append-only machine-readable candidate records.
- `experiments/candidate_registry.csv` — spreadsheet view.
- `experiments/candidate_registry.md` — compact human summary.

## Non-negotiable submission rule

A candidate is not eligible for official submission unless it has:

1. reproducible config/profile/checkpoint/zip paths;
2. zip SHA256 recorded;
3. patch size / chunk size / overlap / stitching recorded;
4. official `starter_code/evaluate.py` result or an explicit documented reason why local official evaluation is not applicable;
5. submission zip check passed;
6. conclusion field filled with the three-line experiment conclusion or a short equivalent.

## Current registry anchor

The current best official artifact is `blend_best075_lir025_20260517` with
score `53.32`. Treat older hard-router and raw-LIR rows as historical evidence
unless a newer registry row explicitly promotes them.

## Legacy router example

The historical `router_t0165` row was generated through the unified entry rather
than the old ad-hoc router script:

```bash
source scripts/env.sh
$PYTHON scripts/unified_predict.py \
  --name router_t0165 \
  --threshold 0.0165 \
  --patch-size 8192 \
  --out-dir results/denoise_router_t0165 \
  --zip result_denoise_router_t0165.zip \
  --append-registry
```

Do not run this historical example locally unless we intentionally recreate the
legacy router baseline. For new formal candidates, prefer the current anchor and
candidate flow documented in `README_REPRODUCE.md` and the registry rows.

## Official evaluation wrapper

For labeled validation directories shaped like:

```text
pred_dir/  shapenet/<category>/<model_id>/denoised.npy
gt_dir/    shapenet/<category>/<model_id>/clean.npy
noisy_dir/ shapenet/<category>/<model_id>/noisy.npy
mesh_dir/  shapenet/<category>/<model_id>/models/model_normalized.obj  # optional for P2S
```

run:

```bash
source scripts/env.sh
$PYTHON scripts/run_official_eval.py \
  --pred-dir <pred_dir> \
  --gt-dir <gt_dir> \
  --noisy-dir <noisy_dir> \
  --mesh-dir <mesh_dir> \
  --out-json experiments/official_eval_<candidate>.json
```

Then append/update the candidate record with:

```bash
$PYTHON scripts/candidate_registry.py \
  --name <candidate> \
  --config <config.yaml> \
  --ckpt <checkpoint.pkl> \
  --zip <result.zip> \
  --branch <branch-name> \
  --patch-size 8192 \
  --official-eval-json experiments/official_eval_<candidate>.json \
  --submission-check passed \
  --conclusion "<three-line conclusion>"
```

## A6000 firewall reminder

Remote execution must obey:

- run `nvidia-smi` first;
- stop immediately on GPU/NVML error;
- no `sudo`, no `apt`, no system pollution;
- work only under `/workspace/freshman`;
- use conda compiler env;
- no concurrent Jittor training/inference jobs.
