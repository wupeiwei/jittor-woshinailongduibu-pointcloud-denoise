# Repository Boundary Audit - 2026-05-19

Purpose: freeze the current repository boundary while the project is still in
active research. This is a Phase A audit only: it does not require deleting,
moving, or rewriting scripts.

## 1. Boundary Rules

- Preserve research velocity: keep experimental scripts available.
- Freeze the mainline: make the recommended train / predict / submit / evaluate
  path explicit.
- Isolate experiment state: `analysis/`, scratch zips, caches, logs, and probe
  outputs are working artifacts, not reproduction source.
- Treat status claims as evidence-backed: official score claims should point to
  `experiments/candidate_registry.*` or a curated report.
- Do not promote smoke/proxy results to official-submission guidance.

## 2. Current Candidate Status

Source of truth: `experiments/candidate_registry.md`,
`experiments/candidate_registry.csv`, and `experiments/candidates.jsonl`.

| Status | Candidate / path | Evidence | Boundary decision |
|---|---|---|---|
| Current best official artifact | `blend_best075_lir025_20260517` / fixed075 | official `53.32`, CD `40.65`, P2S `65.99` in candidate registry | Recommended unless superseded by a newer registry entry |
| Stable VM anchor | `official_vm_fixed_stitch_full_repaired_20260515` and streaming equivalent | official `48.04`; streaming path ties score and validates engineering path | Keep as baseline/reference, not current best |
| Valid but not promoted | `noise_gate_adaptive_blend_20260518` | official `52.57`, below fixed075 `53.32` | Archive as evidence; not recommended for submission |
| Valid but not promoted | `p0_plane_balanced_20260518` | official `53.22`, below fixed075 by `0.10` | Archive as near-neutral P0 evidence |
| Experimental chain check | `garad_identity_fixed075_20260519` | package check OK; movement vs fixed075 exactly zero | Keep as adapter-chain smoke evidence |
| Experimental, not submission recommendation | `garad_bounded_tiny_shrink_a002_20260519` | package/risk pass; no GT/official evidence | Candidate only; do not submit without new evidence |
| Archive / contrast | `garad_bounded_tiny_extend_a002/a005_20260519` | package/risk pass but increases movement vs noisy | Archive as contrast, not recommendation |
| Historical / not promoted | raw LIR and gated LIR variants | official `39.08` / `37.16`, below VM anchor | Keep for diagnosis, not current submission |

## 3. Mainline Entrypoints

These are the entrypoints that should be presented to future readers as the
formal mainline.

| Task | Entrypoint | Notes |
|---|---|---|
| Environment setup | `source scripts/env.sh` | Activates the expected Python environment and compiler wrapper when present |
| Dependency install | `bash scripts/install_deps.sh` | Uses `requirements.txt`; Jittor/CUDA remains machine-sensitive |
| Environment check | `scripts/check_env.py` | Should be split in Phase B into Python/NumPy, Jittor CPU, and Jittor CUDA checks |
| Data check | `scripts/check_data.py` | Pure data-path validation; should stay lightweight |
| Training baseline / ablations | `bash scripts/train.sh <config> [profile]` | Wraps `denoise_baseline.py --mode train` |
| Formal candidate inference / zip | `scripts/unified_predict.py` | Preferred candidate predict / route / zip path |
| Submission zip validation | `scripts/check_submission.py` | Validates formal zip structure, shape, dtype, finiteness, and test-root match |
| Candidate suite | `scripts/evaluate_candidate_suite.py` | Hidden-test-safe artifact checks; not official CD/P2S |
| Official evaluator wrapper | `scripts/run_official_eval.py` | Use only when GT/evaluator inputs are available |
| Candidate registry | `scripts/candidate_registry.py` | Append/rebuild `experiments/candidates.jsonl`, `.csv`, `.md` |

Recommended formal candidate flow:

```bash
source scripts/env.sh
$PYTHON scripts/unified_predict.py --name <candidate> --out-dir <dir> --zip <zip>
$PYTHON scripts/check_submission.py <zip> --test-root dataset_test_noisy --require-float32
$PYTHON scripts/evaluate_candidate_suite.py --candidate <name>=<zip>
$PYTHON scripts/candidate_registry.py --name <candidate> --zip <zip> --submission-check passed --conclusion "<evidence-backed conclusion>"
```

`scripts/predict.sh` remains valid for baseline reproduction, but it should not
be described as the preferred formal-candidate path when `unified_predict.py`
is available.

## 4. Research / Experimental Scripts

These scripts may be valuable but should not be documented as the formal
reproduction path.

| Group | Files | Boundary |
|---|---|---|
| GARA-D matrices and confirms | root `run_garad_v01_*.sh`, `scripts/train_garad_v0.py`, `scripts/train_garad_fixed075_proxy.py`, `scripts/train_garad_vm_base_phase1.py` | Research/proxy training and diagnostics |
| GARA-D artifact adapters | `scripts/garad_identity_adapter.py`, `scripts/garad_bounded_tiny_adapter.py`, `scripts/evaluate_bounded_tiny_proxy_gt.py` | Experimental adapter chain; not mainline submission unless registry promotes it |
| P0 / adaptive blend probes | `scripts/p0_geometry_blend_gate.py`, `scripts/adaptive_blend_probe.py` | Artifact-level experiments; keep conclusions in registry/docs |
| Router / estimator diagnostics | `scripts/predict_router.py`, `scripts/evaluate_router.py`, `scripts/evaluate_noise_estimator.py`, `scripts/summarize_router_stress.py` | Legacy or diagnostic; do not use as new default path |
| Scans / calibration / prototypes | `scripts/scan_*`, `scripts/calibrate_move_scale_refs.py`, `scripts/prototype_move_scale_gate.py`, `scripts/inspect_official_vm_patch_coverage.py` | Engineering/research diagnostics |
| A6000 workflow wrappers | `scripts/a6000_*.sh`, `scripts/activate_a6000_workspace.sh` | Machine-specific workflow; not generic reproduction |
| Smoke / legacy wrappers | `scripts/smoke/*.sh` | Smoke tests and historical wrappers |

Phase B can add `Experimental:` header comments to these scripts, but Phase A
does not require editing them.

## 5. Deprecated / Rejected / Not Promoted Paths

Use these labels to prevent old notes from becoming accidental guidance.

- `fixed075` is the current best artifact, but broad blind blend sweeps are
  research history, not an algorithmic mainline.
- `noise_gate_adaptive_blend_20260518` is valid but below fixed075; do not
  promote unless a newer official result supersedes it.
- `p0_plane_balanced_20260518` is valid but below fixed075; archive as near miss.
- `garad_bounded_tiny_extend_*` variants are archive/contrast only.
- Synthetic VM proxy, smoothing proxy, paired-noise, paired-offset, and fixed075
  proxy runs are diagnostic unless promoted through the candidate registry.
- Legacy `predict_router.py` hard-router candidates are historical; use
  `unified_predict.py` for new candidates.

## 6. Preserved Artifacts

These should be kept or intentionally migrated before any cleanup.

| Artifact | Current location | Decision |
|---|---|---|
| Candidate registry JSONL/CSV/MD | `experiments/candidates.jsonl`, `experiments/candidate_registry.csv`, `experiments/candidate_registry.md` | Preserve as primary candidate index; Phase B should decide whether to unignore or migrate |
| Current best zip metadata | registry row for `blend_best075_lir025_20260517` | Preserve via registry; zip itself can remain external/scratch |
| Official VM fixed-stitch evidence | registry rows and `analysis/large_cloud_pressure_official_vm_fixed_stitch_20260516/` reports | Keep selected report in `docs/experiments/` in Phase C |
| Identity / bounded tiny evidence | `analysis/eval_suite_garad_identity_20260519/`, `analysis/eval_suite_garad_bounded_tiny_20260519/`, adapter risk reports | Keep only selected markdown summaries in Phase C |
| Formal configs | `configs/*.yaml`, `configs/profiles/*.yaml` | Preserve; separate mainline configs from ablation configs in docs |
| Starter VM patch evidence | `starter_code/src/model/vm.py` plus related configs | Document as patched official VM, not untouched starter code |

## 7. Ignored / Scratch Artifacts

Recommended Phase B ignore policy:

- Ignore `analysis/` by default.
- Ignore generated caches: `cache/`, `*.npz`, `*.npy`, `*.pkl`, `*.zip`,
  `*.train.csv`, logs, and temporary run folders.
- Continue ignoring `results/`, large checkpoints, local virtualenvs, and
  dataset symlinks.
- Preserve candidate registry explicitly if it should be versioned, for example
  by unignoring only:

```gitignore
!/experiments/candidates.jsonl
!/experiments/candidate_registry.csv
!/experiments/candidate_registry.md
```

Do not rely on `analysis/` to reproduce results. Reproduction should use
scripts, configs, checkpoints/artifact references, and registry rows.

## 8. Documentation Entry Points

Current documentation should be treated as follows:

| Document | Role | Phase B action |
|---|---|---|
| `README.md` / `README_CN.md` | Public project overview | Add a short "Recommended Entrypoints" table |
| `README_REPRODUCE.md` | Reproduction guide | Separate baseline reproduction from formal candidate pipeline |
| `docs/REPOSITORY_STRUCTURE.md` / `_CN.md` | Repository boundary | Update script lists and note patched starter VM status |
| `docs/CANDIDATE_REGISTRY.md` | Registry workflow | Keep as detailed candidate append guidance |
| `docs/FORMAL_COMPETITION_EXECUTION_CHECKLIST.md` | Competition process | Mark dated/phase-specific conclusions clearly |
| `docs/GPU_PROFILES.md` | Hardware profiles | Keep as hardware guidance, not algorithm guidance |
| `scripts/README_CN.md` | Script grouping | Keep, but align labels with this audit |

## 9. Environment Boundaries

The repository should support three explicit environment levels, not a vague
"works everywhere" promise.

| Level | Expected tools | Should import Jittor? | Examples |
|---|---|---:|---|
| Pure Python / NumPy | Python, NumPy, PyYAML, zipfile | No | `check_submission.py`, parts of `evaluate_candidate_suite.py`, registry bookkeeping |
| Jittor CPU | Jittor import and CPU execution | Yes, CPU only | lightweight smoke/debug when CUDA is unavailable |
| Jittor CUDA | Jittor + CUDA + compiler + CuPy when required | Yes, CUDA | training, denoiser inference, official VM inference |

Known current risks:

- `denoise_baseline.py` imports Jittor at module import time and sets
  `jt.flags.use_cuda = 1` in `main()`.
- `scripts/check_env.py` currently treats CuPy no-device as fatal instead of
  separating CPU and CUDA checks.
- The active venv may still load Jittor from the user site-packages directory;
  this weakens reproducibility.
- `scripts/env.sh` points `JITTOR_HOME` to `.jittor_home/` by default, but
  sandboxed/read-only environments may still need a writable `HOME` before
  importing Jittor.

Phase B should improve error messages and mode separation, not promise universal
one-command portability.

## 10. Phase B Minimal-Safety Checklist

No large cleanup is required before these items:

- Add `.gitignore` coverage for `analysis/` and generated caches while
  preserving candidate registry intentionally.
- Add a recommended-entrypoints table to README / reproduce docs.
- Update repository-structure docs so root-deleted smoke scripts are no longer
  listed as root files; reference `scripts/smoke/` instead.
- Mark research scripts as experimental in docs or headers.
- Split `check_env.py` into clear Python/NumPy, Jittor CPU, and Jittor CUDA
  modes or messages.
- Document that `starter_code` contains patched official VM paths, especially
  fixed-stitch / streaming changes. Current note: `docs/OFFICIAL_VM_PATCHES.md`.

## 11. Phase C Report Migration Candidates

Potential curated reports for `docs/experiments/`:

- `phase0_artifact_audit`: fixed075, VM anchor, raw LIR, noise gate, P0 plane.
- `garad_identity_result`: identity adapter chain and zero-drift validation.
- `bounded_tiny_proxy_eval`: bounded tiny shrink/extend risk and why it is not
  an official recommendation yet.
- `official_vm_fixed_stitch_pressure`: large-cloud fixed-stitch / streaming
  pressure evidence.

Only markdown summaries should move to docs. Large CSV, zip, cache, and NPY
artifacts should remain ignored working outputs.
