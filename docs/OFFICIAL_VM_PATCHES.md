# Official VM Patch Notes

This document explains how `starter_code/src/model/vm.py` differs from the
original official starter-code VM path.

The goal is to make the patched boundary explicit: future readers should know
which behavior is official, which behavior is a repaired official path, and
which behavior is experimental.

## Current Patch Groups

| Patch group | Files | Default behavior | Status |
|---|---|---|---|
| Fixed-stitch coverage repair | `starter_code/src/model/vm.py` | Active in `patch_based_denoise` | Engineering bugfix / should preserve point count |
| Streaming fixed-stitch assignment | `starter_code/src/model/vm.py` | Active for large clouds when `VM_STITCHING=auto`; forced by `VM_STITCHING=streaming`; disabled by `VM_STITCHING=dense` | Engineering patch validated as score-neutral in registry, but should stay in a separate VM patch commit |
| Distance / magnitude gate | `starter_code/src/model/vm.py`, `starter_code/configs/model/vm.yaml`, `starter_code/configs/model/vm_distance_gate.yaml`, related task/system configs | Off in `vm.yaml`; on only in `vm_distance_gate` configs | Experimental StraightPCF-style branch, not part of current official best |
| Quick predict config edits | `starter_code/configs/task/predict_vm_quick.yaml` | Changes local quick workflow target checkpoint/output path | Local workflow convenience; keep separate from VM algorithm patch |

## 1. Fixed-Stitch Coverage Repair

Original risk:

- the official patch stitching assumes every input point appears in at least
  one FPS/KNN patch;
- when a point is uncovered, the final concatenation can silently drop that
  point or fail to return one output point per input point.

Patch:

- detect uncovered input point ids after the initial KNN patch construction;
- add KNN patches centered at uncovered points;
- assert the final output point count equals the input point count;
- use the original noisy point as a defensive fallback if a stitched assignment
  is unexpectedly empty.

Effect:

- changes inference stitching behavior;
- does not change model weights or checkpoint format;
- is required for robust formal submission artifacts because output count must
  match input count.

## 2. Streaming Fixed-Stitch Assignment

Original risk:

- the dense fixed-stitch assignment builds `all_dists(num_patches, N)`;
- large clouds can hit a memory cliff, especially around synthetic 500k-point
  pressure tests.

Patch:

- add `_streaming_best_assignment(...)`;
- add `patch_based_denoise_streaming(...)`;
- keep FPS seeds, KNN patches, normalization, checkpoint, and Langevin denoising
  unchanged;
- replace the dense `all_dists(P,N)` storage with O(N) best-patch arrays;
- route in `predict_step` by environment variable:

```text
VM_STITCHING=auto       # default, streaming for N >= VM_STREAMING_MIN_POINTS
VM_STITCHING=streaming  # force streaming path
VM_STITCHING=dense      # force dense repaired path
VM_STREAMING_MIN_POINTS # default 32768
```

Evidence:

- curated report: `docs/experiments/official_vm_fixed_stitch_pressure_20260516.md`;
- source analysis reports under
  `analysis/large_cloud_pressure_official_vm_fixed_stitch_20260516/`;
- candidate registry records `official_vm_streaming_200_smoke_20260516` as
  official score `48.04`, tying the repaired fixed-stitch reference.

Effect:

- this is an engineering patch for memory/point-count safety;
- it should be committed separately from baseline model research changes;
- it is not a score-improvement claim by itself.

## 3. Distance / Magnitude Gate

Patch:

- add optional `distance_estimation`, `distance_gate`, and `distance_gate_min`
  config fields;
- pass `distance_estimation` into `FeatureExtraction`;
- add a small `distance_decoder` and multiply predicted direction by the gate
  when enabled.

Default:

- `starter_code/configs/model/vm.yaml` sets these flags to `false`;
- existing official VM checkpoints should behave as before unless a
  `vm_distance_gate` config is explicitly selected.

Status:

- experimental StraightPCF-style research branch;
- should not be mixed into an "official VM fixed-stitch bugfix" commit;
- should be reviewed separately before promotion.

## 4. Recommended Commit Boundary

Recommended split:

1. `smoke script relocation`
2. `repo-boundary-docs-and-env`
3. `candidate registry + docs/experiments`
4. `README_REPRODUCE legacy/router wording`
5. `baseline/model research changes`
6. `official VM patch changes`
7. `experimental VM distance-gate configs`

The VM patch commit should include fixed-stitch coverage repair and streaming
assignment only. Distance-gate changes should be kept separate unless the commit
is explicitly labeled experimental.
