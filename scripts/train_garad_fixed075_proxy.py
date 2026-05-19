#!/usr/bin/env python3
"""GARA-D v0.1 fixed075-style pseudo-residual diagnostic.

Phase 1.5 goal:
  Build a train-split diagnostic whose base distribution is closer to the current
  fixed075 hidden-test blend than the toy smoothing base.

Pipeline:
  clean OBJ -> sampled clean -> synthetic noisy
  stream proxy = trained local ResidualDenoiser prediction(noisy)
  lir proxy    = two-step iterative refinement with the same denoiser
  fixed075 base = 0.75 * stream + 0.25 * lir
  adapter target = clean - fixed075_base

This is still train-split synthetic diagnostic, not an official score estimate.
It must not use hidden-test labels.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

_early = argparse.ArgumentParser(add_help=False)
_early.add_argument("--cpu", action="store_true")
_early_args, _ = _early.parse_known_args()
if _early_args.cpu:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("DISABLE_MULTIPROCESSING", "1")
    os.environ["nvcc_path"] = ""

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# `denoise_baseline.py` imports trimesh at module import time for its own OBJ
# loader, but this diagnostic uses the lightweight OBJ reader from
# scripts.train_garad_v0 instead. Avoid installing/depending on trimesh just to
# import ResidualDenoiser.
try:
    import trimesh  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    import types
    sys.modules["trimesh"] = types.SimpleNamespace(load=None)

import jittor as jt
from jittor import nn

from denoise_baseline import ResidualDenoiser, predict_points_in_chunks
from scripts.train_garad_v0 import (
    GARADv0,
    ResidualMLPAdapter,
    ZeroAdapter,
    chamfer_l2_jt,
    chamfer_l2_np,
    geometry_features_np,
    list_obj_files,
    load_obj_vertices,
    metric_to_score,
    normalize_pc,
    sample_points_rng,
    scalar,
    set_seed,
    smooth_base_prediction,
)


def repo_path(path: str | Path | None) -> Path | None:
    if path is None or str(path) == "":
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def build_base_model(args: argparse.Namespace) -> ResidualDenoiser:
    model = ResidualDenoiser(
        k=args.base_model_k,
        feat_dim=args.base_model_feat_dim,
        hidden=args.base_model_hidden,
        use_pwsenel=args.base_model_pwsenel,
        use_staas=args.base_model_staas,
        staas_strength=args.base_model_staas_strength,
        use_move_gate=args.base_model_move_gate,
        use_pwsenel_v2=args.base_model_pwsenel_v2,
        residual_clip=args.base_model_residual_clip,
        adaptive_clip=args.base_model_adaptive_clip,
        noise_aware_move_gate=args.base_model_noise_aware_move_gate,
        hybrid_safe_strong=args.base_model_hybrid_safe_strong,
        pwsenel_v2_edge_lock=args.base_model_pwsenel_v2_edge_lock,
        pwsenel_v2_gate_scale=args.base_model_pwsenel_v2_gate_scale,
        adaptive_clip_min=args.base_model_adaptive_clip_min,
        adaptive_clip_mid=args.base_model_adaptive_clip_mid,
        adaptive_clip_max=args.base_model_adaptive_clip_max,
        adaptive_clip_ref_low=args.base_model_adaptive_clip_ref_low,
        adaptive_clip_ref_mid=args.base_model_adaptive_clip_ref_mid,
        adaptive_clip_ref_high=args.base_model_adaptive_clip_ref_high,
        noise_aware_gate_min=args.base_model_noise_aware_gate_min,
        noise_aware_gate_ref_low=args.base_model_noise_aware_gate_ref_low,
        noise_aware_gate_ref_high=args.base_model_noise_aware_gate_ref_high,
        hybrid_router_scale=args.base_model_hybrid_router_scale,
    )
    model.load(str(repo_path(args.base_ckpt)))
    model.eval()
    return model


def fixed075_base_from_model(model: ResidualDenoiser, noisy: np.ndarray, patch_size: int, lir_steps: int, lir_alpha: float, stream_weight: float = 0.75, smooth_mix: float = 0.0, smooth_k: int = 8, smooth_alpha: float = 0.04) -> tuple[np.ndarray, dict[str, float]]:
    stream = predict_points_in_chunks(model, noisy.astype(np.float32), patch_size).astype(np.float32)
    current = noisy.astype(np.float32)
    step_delta = []
    for _ in range(lir_steps):
        pred = predict_points_in_chunks(model, current, patch_size).astype(np.float32)
        step_delta.append(float(np.sqrt(((pred - current) ** 2).sum(axis=1)).mean()))
        current = ((1.0 - lir_alpha) * current + lir_alpha * pred).astype(np.float32)
    lir = current
    blend = (stream_weight * stream + (1.0 - stream_weight) * lir).astype(np.float32)
    if smooth_mix > 0:
        smooth = smooth_base_prediction(noisy.astype(np.float32), k=smooth_k, alpha=smooth_alpha).astype(np.float32)
        base = ((1.0 - smooth_mix) * blend + smooth_mix * smooth).astype(np.float32)
    else:
        base = blend
    return base, {
        "stream_weight": float(stream_weight),
        "smooth_mix": float(smooth_mix),
        "stream_lir_disp_mean": float(np.sqrt(((stream - lir) ** 2).sum(axis=1)).mean()),
        "stream_lir_disp_p95": float(np.quantile(np.sqrt(((stream - lir) ** 2).sum(axis=1)), 0.95)),
        "lir_step_delta_mean": float(np.mean(step_delta)) if step_delta else 0.0,
    }


def make_cached_sample(sample: str, obj: Path, idx: int, split_seed: int, args: argparse.Namespace, base_model: ResidualDenoiser) -> dict[str, Any]:
    rng = np.random.default_rng(split_seed + idx * 1009)
    clean_full = normalize_pc(load_obj_vertices(obj))
    clean = sample_points_rng(clean_full, args.num_points, rng)
    sigma = float(rng.uniform(args.noise_min, args.noise_max))
    noisy = clean + rng.normal(0.0, sigma, clean.shape).astype(np.float32)
    base, base_diag = fixed075_base_from_model(
        base_model,
        noisy,
        args.base_predict_patch_size,
        args.lir_steps,
        args.lir_alpha,
        args.stream_weight,
        args.smooth_mix,
        args.smooth_k,
        args.smooth_alpha,
    )
    geom = geometry_features_np(noisy, base, k=args.geom_k)
    x = np.concatenate([noisy, base, noisy - base, geom], axis=1).astype(np.float32)
    return {
        "sample": sample,
        "clean": clean.astype(np.float32),
        "noisy": noisy.astype(np.float32),
        "base": base.astype(np.float32),
        "x": x,
        "sigma": np.asarray([sigma], dtype=np.float32),
        **base_diag,
    }


def build_or_load_cache(split: str, samples: list[tuple[str, Path]], split_seed: int, args: argparse.Namespace, out_dir: Path) -> list[Path]:
    cache_dir = out_dir / "cache" / split
    cache_dir.mkdir(parents=True, exist_ok=True)
    expected = [cache_dir / f"{i:04d}.npz" for i in range(len(samples))]
    if all(p.exists() for p in expected):
        return expected
    print(f"building {split} fixed075-style cache: {len(samples)} samples", flush=True)
    base_model = build_base_model(args)
    rows = []
    for i, (sample, obj) in enumerate(samples):
        p = expected[i]
        item = make_cached_sample(sample, obj, i, split_seed, args, base_model)
        np.savez_compressed(
            p,
            clean=item["clean"], noisy=item["noisy"], base=item["base"], x=item["x"], sigma=item["sigma"], sample=np.asarray(item["sample"]),
        )
        cd_noisy = chamfer_l2_np(item["noisy"], item["clean"])
        cd_base = chamfer_l2_np(item["base"], item["clean"])
        row = {
            "idx": i,
            "sample": sample,
            "sigma": float(item["sigma"][0]),
            "cd_noisy": cd_noisy,
            "cd_base": cd_base,
            "base_better_than_noisy": int(cd_base < cd_noisy),
            "stream_lir_disp_mean": item["stream_lir_disp_mean"],
            "stream_lir_disp_p95": item["stream_lir_disp_p95"],
            "lir_step_delta_mean": item["lir_step_delta_mean"],
        }
        rows.append(row)
        print(f"[{split} {i+1}/{len(samples)}] cd_noisy={cd_noisy:.8g} cd_base={cd_base:.8g}", flush=True)
    with (cache_dir / "cache_diagnostics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return expected


def build_adapter(args: argparse.Namespace) -> nn.Module:
    if args.model == "garad":
        return GARADv0(in_dim=14, hidden=args.hidden, max_step=args.max_step)
    if args.model == "residual_mlp":
        return ResidualMLPAdapter(in_dim=14, hidden=args.hidden, max_step=args.max_step)
    if args.model == "zero":
        return ZeroAdapter()
    raise ValueError(args.model)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in ["x", "base", "clean", "noisy", "sigma"]}


def make_batch(cache: list[Path], args: argparse.Namespace) -> tuple[jt.Var, jt.Var, jt.Var, jt.Var]:
    xs, bases, cleans, noisys = [], [], [], []
    for _ in range(args.batch_size):
        s = load_npz(random.choice(cache))
        xs.append(s["x"]); bases.append(s["base"]); cleans.append(s["clean"]); noisys.append(s["noisy"])
    return jt.array(np.stack(xs)), jt.array(np.stack(bases)), jt.array(np.stack(cleans)), jt.array(np.stack(noisys))


def evaluate(model: nn.Module, cache: list[Path], args: argparse.Namespace, out_csv: Path) -> dict[str, Any]:
    rows = []
    for i, p in enumerate(cache[: args.eval_limit], 1):
        s = load_npz(p)
        with jt.no_grad():
            pred_j, delta_j, gate_j, dist_j = model(jt.array(s["x"][None, ...]), jt.array(s["base"][None, ...]), return_delta=True)
        pred = np.asarray(pred_j.numpy()[0], dtype=np.float32)
        delta = np.asarray(delta_j.numpy()[0], dtype=np.float32)
        gate = np.asarray(gate_j.numpy()[0], dtype=np.float32)
        dist = np.asarray(dist_j.numpy()[0], dtype=np.float32)
        cd_noisy = chamfer_l2_np(s["noisy"], s["clean"])
        cd_base = chamfer_l2_np(s["base"], s["clean"])
        cd_pred = chamfer_l2_np(pred, s["clean"])
        rows.append({
            "idx": i,
            "cache": str(p),
            "sigma": float(s["sigma"][0]),
            "cd_noisy": cd_noisy,
            "cd_base": cd_base,
            "cd_pred": cd_pred,
            "base_better_than_noisy": int(cd_base < cd_noisy),
            "pred_better_than_base": int(cd_pred < cd_base),
            "pred_better_than_noisy": int(cd_pred < cd_noisy),
            "score_vs_base": metric_to_score(cd_pred, cd_base),
            "delta_l2_mean": float(np.sqrt((delta * delta).sum(axis=1)).mean()),
            "delta_l2_p95": float(np.quantile(np.sqrt((delta * delta).sum(axis=1)), 0.95)),
            "gate_mean": float(gate.mean()),
            "distance_mean": float(dist.mean()),
        })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    means = {k: float(np.mean([r[k] for r in rows])) for k in ["cd_noisy", "cd_base", "cd_pred", "score_vs_base", "delta_l2_mean", "delta_l2_p95", "gate_mean", "distance_mean"]}
    means["pred_better_than_base_rate"] = float(np.mean([r["pred_better_than_base"] for r in rows]))
    means["base_better_than_noisy_rate"] = float(np.mean([r["base_better_than_noisy"] for r in rows]))
    means["n"] = len(rows)
    return means


def main() -> None:
    p = argparse.ArgumentParser(description="Train/evaluate GARA-D on fixed075-style pseudo-residual cache.")
    p.add_argument("--data-root", default="dataset_train")
    p.add_argument("--train-list", default="starter_code/datalist/train.txt")
    p.add_argument("--eval-list", default="starter_code/datalist/validate.txt")
    p.add_argument("--out-dir", default="analysis/garad_v01_fixed075_proxy_20260519")
    p.add_argument("--seed", type=int, default=20260519)
    p.add_argument("--train-limit", type=int, default=32)
    p.add_argument("--eval-limit", type=int, default=32)
    p.add_argument("--steps", type=int, default=160)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-points", type=int, default=512)
    p.add_argument("--noise-min", type=float, default=0.005)
    p.add_argument("--noise-max", type=float, default=0.02)
    p.add_argument("--geom-k", type=int, default=16)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--model", choices=["garad", "residual_mlp", "zero"], default="garad")
    p.add_argument("--max-step", type=float, default=0.010)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--cd-num-points", type=int, default=512)
    p.add_argument("--lambda-offset", type=float, default=0.05)
    p.add_argument("--lambda-delta", type=float, default=1.0)
    p.add_argument("--lambda-cd", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--base-ckpt", default="experiments/denoise_baseline/baseline.pkl")
    p.add_argument("--base-name", default="baseline")
    p.add_argument("--base-predict-patch-size", type=int, default=1000)
    p.add_argument("--lir-steps", type=int, default=2)
    p.add_argument("--lir-alpha", type=float, default=0.75)
    p.add_argument("--stream-weight", type=float, default=0.75)
    p.add_argument("--smooth-mix", type=float, default=0.0)
    p.add_argument("--smooth-k", type=int, default=8)
    p.add_argument("--smooth-alpha", type=float, default=0.04)
    p.add_argument("--base-model-k", type=int, default=16)
    p.add_argument("--base-model-feat-dim", type=int, default=256)
    p.add_argument("--base-model-hidden", type=int, default=256)
    p.add_argument("--base-model-pwsenel", action="store_true")
    p.add_argument("--base-model-staas", action="store_true")
    p.add_argument("--base-model-staas-strength", type=float, default=1.0)
    p.add_argument("--base-model-move-gate", action="store_true")
    p.add_argument("--base-model-pwsenel-v2", action="store_true")
    p.add_argument("--base-model-pwsenel-v2-edge-lock", type=float, default=0.7)
    p.add_argument("--base-model-pwsenel-v2-gate-scale", type=float, default=0.5)
    p.add_argument("--base-model-residual-clip", type=float, default=0.0)
    p.add_argument("--base-model-adaptive-clip", action="store_true")
    p.add_argument("--base-model-adaptive-clip-min", type=float, default=0.006)
    p.add_argument("--base-model-adaptive-clip-mid", type=float, default=0.010)
    p.add_argument("--base-model-adaptive-clip-max", type=float, default=0.020)
    p.add_argument("--base-model-adaptive-clip-ref-low", type=float, default=0.022)
    p.add_argument("--base-model-adaptive-clip-ref-mid", type=float, default=0.030)
    p.add_argument("--base-model-adaptive-clip-ref-high", type=float, default=0.040)
    p.add_argument("--base-model-noise-aware-move-gate", action="store_true")
    p.add_argument("--base-model-noise-aware-gate-min", type=float, default=0.45)
    p.add_argument("--base-model-noise-aware-gate-ref-low", type=float, default=0.022)
    p.add_argument("--base-model-noise-aware-gate-ref-high", type=float, default=0.036)
    p.add_argument("--base-model-hybrid-safe-strong", action="store_true")
    p.add_argument("--base-model-hybrid-router-scale", type=float, default=1.0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    jt.flags.use_cuda = 0 if args.cpu else 1
    out_dir = repo_path(args.out_dir); data_root = repo_path(args.data_root); train_list = repo_path(args.train_list); eval_list = repo_path(args.eval_list)
    assert out_dir and data_root and train_list and eval_list
    out_dir.mkdir(parents=True, exist_ok=True)

    train_samples = list_obj_files(data_root, train_list, args.train_limit)
    eval_samples = list_obj_files(data_root, eval_list, args.eval_limit)
    train_cache = build_or_load_cache("train", train_samples, args.seed, args, out_dir)
    eval_cache = build_or_load_cache("eval", eval_samples, args.seed + 9999, args, out_dir)

    model = build_adapter(args)
    params = list(model.parameters()) if hasattr(model, "parameters") else []
    opt = nn.Adam(params, lr=args.lr) if params else None
    log_path = out_dir / "train_log.csv"
    with log_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "loss", "loss_cd", "loss_delta", "loss_offset", "delta_l2_mean", "gate_mean", "distance_mean", "elapsed_sec"])
        w.writeheader(); t0 = time.time()
        for step in range(1, args.steps + 1):
            x, base, clean, _noisy = make_batch(train_cache, args)
            pred, delta, gate, dist = model(x, base, return_delta=True)
            loss_cd = chamfer_l2_jt(pred, clean, args.cd_num_points)
            delta_l2 = ((delta ** 2).sum(dim=-1) + 1e-12) ** 0.5
            loss_offset = (delta_l2 ** 2).mean()
            loss_delta = ((delta - (clean - base)) ** 2).mean()
            loss = args.lambda_cd * loss_cd + args.lambda_delta * loss_delta + args.lambda_offset * loss_offset
            if opt is not None:
                opt.step(loss)
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                row = {"step": step, "loss": scalar(loss), "loss_cd": scalar(loss_cd), "loss_delta": scalar(loss_delta), "loss_offset": scalar(loss_offset), "delta_l2_mean": scalar(delta_l2.mean()), "gate_mean": scalar(gate.mean()), "distance_mean": scalar(dist.mean()), "elapsed_sec": round(time.time() - t0, 3)}
                w.writerow(row); f.flush(); print(row, flush=True)

    ckpt = out_dir / f"{args.model}_v0.pkl"
    if params:
        model.save(str(ckpt))
    else:
        ckpt.write_text("zero adapter has no parameters\n")
    eval_summary = evaluate(model, eval_cache, args, out_dir / "eval_fixed075_proxy.csv")
    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "purpose": "GARA-D fixed075-style pseudo-residual diagnostic; train split synthetic only",
        "args": vars(args),
        "train_cache": len(train_cache),
        "eval_cache": len(eval_cache),
        "checkpoint": str(ckpt),
        "eval": eval_summary,
        "pass_gate": bool(eval_summary["cd_pred"] < eval_summary["cd_base"] and eval_summary["pred_better_than_base_rate"] >= 0.60),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    report = [
        "# GARA-D fixed075-style pseudo-residual diagnostic",
        "",
        f"Train-split synthetic diagnostic only. Base recipe: `{args.base_name}`, stream_weight={args.stream_weight}, smooth_mix={args.smooth_mix}.",
        "",
        f"- eval n: {eval_summary['n']}",
        f"- cd_noisy: {eval_summary['cd_noisy']:.8g}",
        f"- cd_base: {eval_summary['cd_base']:.8g}",
        f"- cd_pred: {eval_summary['cd_pred']:.8g}",
        f"- base_better_than_noisy_rate: {eval_summary['base_better_than_noisy_rate']:.3f}",
        f"- pred_better_than_base_rate: {eval_summary['pred_better_than_base_rate']:.3f}",
        f"- delta_l2_mean: {eval_summary['delta_l2_mean']:.8g}",
        f"- gate_mean: {eval_summary['gate_mean']:.8g}",
        "",
        f"Pass gate: `{summary['pass_gate']}` (`cd_pred < cd_base` and win_rate >= 0.60).",
    ]
    (out_dir / "risk_report.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"out_dir": str(out_dir), "eval": eval_summary, "pass_gate": summary["pass_gate"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
