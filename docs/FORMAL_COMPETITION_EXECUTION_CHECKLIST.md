# ARCHIVE - Formal Competition Execution Checklist

Historical status: this document records the execution plan and decisions from
an earlier competition phase. It is not the current repository boundary or
submission recommendation. For current status, read
`docs/repo_boundary_audit_20260519.md` and `experiments/candidate_registry.md`.

Historical source plan omitted.

Status: canonical v2 frozen at 2026-05-08 21:42 GMT+8. This checklist follows the formal competition plan, not the standalone ST-AAS innovation plan.

## Non-negotiable principle

```text
比赛工程要同构，B榜默认纯推理，SOTA 借鉴要提前，创新模块要消融，评估必须用官方口径，router 要连续稳健，阈值调参只能辅助，答辩复现从第一天开始准备。
```

## Phase 1: now to mid-May

Goal: engineering closure + router stabilization + official baseline/StraightPCF + official evaluate.py + automatic Candidate Registry + UnifiedDenoisePipeline prototype.

Ordered tasks:

1. [x] Submit and record `router_t0165` official result.
   - Official result: `score=35.91`, `CD_score=26.95`, `P2S_score=44.87`.
   - Decision: below stable `router_t016=39.92`; keep as hard-router ablation/archive, not as current best.
2. [x] Decide `t01625` or `t017` from `t0165` result.
   - Decision: do **not** continue threshold sweep as the main line; move to official baseline / StraightPCF / official evaluate.py / soft-gate stress per canonical plan.
3. [x] Read and run official baseline / StraightPCF.
   - Official baseline quick chain is already locally runnable: `starter_code/experiments/vm_quick/checkpoint_0.pkl` and one checked `denoised.npy` output.
   - Caveat: quick output path currently has an extra `dataset_test_noisy` nesting; keep as sanity baseline, not a submission package.
   - StraightPCF source still needs reduced-scope code reading; do not port full PyTorch project blindly.
4. [x] Add official baseline / StraightPCF as first Candidate Registry baselines.
   - Added `official_vm_quick` as first official baseline sanity row.
   - StraightPCF remains a reference baseline/idea source until a Jittor-compatible minimal prototype exists.
5. [x] Build automatic Candidate Registry skeleton:
   - `experiments/candidates.jsonl`
   - `experiments/candidate_registry.csv`
   - `experiments/candidate_registry.md`
   - writer: `scripts/candidate_registry.py`
6. [x] Add official `starter_code/evaluate.py` wrapper for CD/P2S registry capture: `scripts/run_official_eval.py`.
7. [x] Fix low/mid/high/overall bucket evaluation with same seed, count, script version, and patch size.
   - Baseline bucket eval completed on A6000: `analysis/phase1_bucket_eval_baseline_20260512_2009/summary.md`.
   - Key result: low-noise bucket is the failure bucket (`mean_cd_score=14.4883`, `median_cd_score=0`, `pred_better_rate=45.3%`), while mid/high buckets are stable.
8. [x] Fix patch size recording; current preferred value: `8192`.
9. [x] Move router into formal `UnifiedDenoisePipeline` / `NoisyConditionedDenoiser` path skeleton: `scripts/unified_predict.py`.
10. [x] Compare hard router vs soft gate and run wrong-route stress tests.
   - Router stress modes (`hard`, `soft`, `force-safe`, `force-strong`) were made runnable through `scripts/unified_predict.py`.
   - Synthetic GT limit-64 + threshold sweep showed `force-strong > soft > hard > force-safe` under Gaussian synthetic CD, but this did **not** transfer to hidden official test.
   - Official force-strong full submission scored `33.46` (`CD_score=21.67`, `P2S_score=45.25`), far below `official_vm_fixed_stitch/full_repaired=48.04`; conclusion: treat router/strong as diagnostic evidence, not current submission priority.
11. [x] Read SOTA code only in the reduced scope: StraightPCF / official baseline + U-CAN.
   - StraightPCF official code read from `/home/sallen/.openclaw/workspace/external_refs/straightpcf` commit `223b240`.
   - Transferable idea: bounded distance/move-scale head + straight residual trajectory; avoid direct PyTorch/PyG/PyTorch3D port.
   - U-CAN currently treated as paper-level inspiration for multi-step consistency / Noise2Noise matching; no obvious official code entry found in quick source check.
12. [x] Build Jittor operator compatibility prototypes for future SOTA-inspired modules.
    - 2026-05-12: Added standalone `scripts/prototype_move_scale_gate.py` for StraightPCF-style bounded `move_scale / distance gate`. A6000 smoke validated forward, KNN gather, bounded scale, loss, backward, and Adam step. Caveat: raw KNN spacing refs `0.010~0.030` saturate on 2048-point surface patches (`cloud_dist≈0.121~0.128`); refs require real-data calibration before wiring into main denoiser.
    - 2026-05-12: Added real-data calibration via `scripts/calibrate_move_scale_refs.py` (`analysis/move_scale_calibration_20260512/`). Synthetic train low/mid/high and official test noisy stats suggest PCA roughness (`plane_res_mean/median`) is a better noise proxy than raw KNN spacing; initial refs: `plane_res_mean low≈0.005 high≈0.0095` or `plane_res_median low≈0.0047 high≈0.0090`.
13. [x] Add official schedule, A/B isomorphism, B-list data firewall, debate 35%, patch/chunk/P2S/speed/memory/SHA256 checks into submission decision logic and docs: `docs/CANDIDATE_REGISTRY.md`.

## Phase 2: mid-May to early June

Goal: decide main architecture candidate.

1. [ ] Implement LIR-Denoiser v0.
2. [ ] T=2 smoke.
3. [ ] Compare against A6000 baseline / noise-aware / router.
4. [ ] Add PW-SENEL / ST-AAS v0 as minimal ablations inside the unified pipeline.
5. [ ] Precheck LIR + ST-AAS coupling:
   - edge_conf frame-to-frame variance;
   - recompute vs first-frame reuse;
   - T=2/3 latency and memory;
   - CD/P2S, offset magnitude, gate variance.
6. [ ] Decide whether to enter long training.

## Phase 3: early June to end of June

Goal: freeze main structure.

1. [ ] Long training.
2. [ ] Full ablation.
3. [ ] CD/P2S optimization.
4. [ ] Patch/chunk large-cloud stability.
5. [ ] 10w / 50w synthetic large-cloud stress tests.
6. [ ] Overlap patch / weighted stitching / boundary consistency.
7. [ ] Fix ST-AAS edge_conf strategy if used.
8. [ ] Clean-env reproduction rehearsal.
9. [ ] A-list candidate pool and reproducibility docs.

## Phase 4: July

Goal: robust optimization, no major refactor.

Allowed: threshold/gate temperature calibration, patch/chunk/overlap/stitching tuning, training steps, checkpoint selection, small config/loss adjustment, bugfix, submission strategy, full A-list candidate reproduction, debate material.

Forbidden: major architecture changes, temporary SOTA swap, unified pipeline rollback, irreproducible postprocess.

## Phase 5: B-list 10 days

Goal: same-code pure inference adaptation.

Allowed: same-code inference, threshold/gate temperature tuning, patch/chunk/overlap/stitching tuning, speed/memory/I/O/zip fixes, noisy-only calibration.

Forbidden: full retrain, finetune, linear probing, BN/running stats update, gradient backprop, different model, backbone rewrite, inference pipeline rewrite, route change, using B-list feedback as implicit labels.

## Current correction

The previous `docs/ST_AAS_EXECUTION_CHECKLIST.md` was created from the innovation plan and is not the formal competition execution order. Keep ST-AAS work as Phase 2 ablation unless needed by the unified pipeline; do not let it replace Phase 1 tasks.
