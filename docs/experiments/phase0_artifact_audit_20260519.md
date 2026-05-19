# Phase 0 Artifact Audit

Source cluster:

- `analysis/eval_suite_20260519_phase0/risk_report.md`
- `analysis/eval_suite_20260519_phase0/summary.json`

## What this covered

The Phase 0 suite checked the current internal reference candidates without
claiming new official metrics.

## Key evidence

| candidate | status | official | note |
|---|---|---:|---|
| `official_vm_streaming` | OK | 48.04 | VM / official fixed-stitch streaming anchor |
| `raw_lir` | OK | 39.08 | raw LIR T=2 alpha=0.75 |
| `fixed075` | OK | 53.32 | current internal best |
| `noise_gate` | OK | 52.57 | valid, below fixed075 |
| `p0_plane_balanced` | OK | 53.22 | valid, near miss |

## Boundary decision

- Keep `blend_best075_lir025_20260517` as the current best official reference.
- Keep the VM anchor and raw LIR as comparison baselines only.
- Keep `noise_gate` and `p0_plane_balanced` as evidence, not promotion.

## Why this matters

This suite is the cleanest short summary of the current official-score boundary
before any new candidate work.
