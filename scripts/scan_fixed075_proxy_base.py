#!/usr/bin/env python3
"""Scan fixed075-style proxy base health on train-split synthetic noisy data.

This script trains no adapter. It only answers whether a candidate base
construction is healthy enough before GARA-D residual training.

Base constructions:
  stream = one-shot local checkpoint prediction(noisy)
  lir    = iterative refinement using the same checkpoint
  base   = w * stream + (1-w) * lir

Optional smoothing blend can mix a cheap noisy-only smoothing base into the
proxy, but the primary goal is to find checkpoint/LIR settings with
base_better_than_noisy_rate >= target.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

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

try:
    import trimesh  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    import types
    sys.modules["trimesh"] = types.SimpleNamespace(load=None)

import jittor as jt

from denoise_baseline import ResidualDenoiser, predict_points_in_chunks
from scripts.train_garad_v0 import chamfer_l2_np, list_obj_files, load_obj_vertices, normalize_pc, sample_points_rng, smooth_base_prediction


@dataclass(frozen=True)
class CheckpointSpec:
    name: str
    ckpt: str
    flags: dict[str, object]


def repo_path(path: str | Path | None) -> Path | None:
    if path is None or str(path) == "":
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def candidate_specs() -> list[CheckpointSpec]:
    return [
        CheckpointSpec("baseline", "experiments/denoise_baseline/baseline.pkl", {}),
        CheckpointSpec("pwsenel_v2_clip", "experiments/denoise_pwsenel_v2_clip/pwsenel_v2_clip.pkl", {"use_pwsenel_v2": True, "pwsenel_v2_edge_lock": 0.7, "pwsenel_v2_gate_scale": 0.5, "residual_clip": 0.01}),
        CheckpointSpec("noise_aware_move_gate", "experiments/denoise_noise_aware_move_gate/noise_aware_move_gate.pkl", {"use_move_gate": True, "noise_aware_move_gate": True, "noise_aware_gate_min": 0.45, "noise_aware_gate_ref_low": 0.022, "noise_aware_gate_ref_high": 0.036}),
        CheckpointSpec("hybrid_safe_strong", "experiments/denoise_hybrid_safe_strong/hybrid_safe_strong.pkl", {"use_pwsenel_v2": True, "pwsenel_v2_edge_lock": 0.7, "pwsenel_v2_gate_scale": 0.5, "use_move_gate": True, "hybrid_safe_strong": True, "hybrid_router_scale": 1.0, "adaptive_clip": True, "adaptive_clip_min": 0.006, "adaptive_clip_mid": 0.012, "adaptive_clip_max": 0.028, "adaptive_clip_ref_low": 0.022, "adaptive_clip_ref_mid": 0.028, "adaptive_clip_ref_high": 0.036}),
    ]


def build_model(spec: CheckpointSpec) -> ResidualDenoiser:
    ckpt = repo_path(spec.ckpt)
    if ckpt is None or not ckpt.exists():
        raise FileNotFoundError(spec.ckpt)
    kwargs = {
        "k": 16,
        "feat_dim": 256,
        "hidden": 256,
        "use_pwsenel": False,
        "use_staas": False,
        "staas_strength": 1.0,
        "use_move_gate": False,
        "use_pwsenel_v2": False,
        "residual_clip": 0.0,
        "adaptive_clip": False,
        "noise_aware_move_gate": False,
        "hybrid_safe_strong": False,
    }
    kwargs.update(spec.flags)
    model = ResidualDenoiser(**kwargs)
    model.load(str(ckpt))
    model.eval()
    return model


def make_clean_noisy(obj: Path, n: int, noise_min: float, noise_max: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float]:
    clean_full = normalize_pc(load_obj_vertices(obj))
    clean = sample_points_rng(clean_full, n, rng)
    sigma = float(rng.uniform(noise_min, noise_max))
    noisy = clean + rng.normal(0.0, sigma, clean.shape).astype(np.float32)
    return clean.astype(np.float32), noisy.astype(np.float32), sigma


def model_outputs(model: ResidualDenoiser, noisy: np.ndarray, patch_size: int, lir_steps: int, lir_alpha: float) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    stream = predict_points_in_chunks(model, noisy, patch_size).astype(np.float32)
    current = noisy.astype(np.float32)
    deltas = []
    for _ in range(lir_steps):
        pred = predict_points_in_chunks(model, current, patch_size).astype(np.float32)
        deltas.append(float(np.sqrt(((pred - current) ** 2).sum(axis=1)).mean()))
        current = ((1.0 - lir_alpha) * current + lir_alpha * pred).astype(np.float32)
    return stream, current, {"lir_step_delta_mean": float(np.mean(deltas)) if deltas else 0.0}


def eval_rows_for_spec(spec: CheckpointSpec, model: ResidualDenoiser, samples: list[dict[str, object]], args: argparse.Namespace) -> list[dict[str, object]]:
    rows = []
    for sample_i, s in enumerate(samples):
        clean = s["clean"]  # type: ignore[assignment]
        noisy = s["noisy"]  # type: ignore[assignment]
        assert isinstance(clean, np.ndarray) and isinstance(noisy, np.ndarray)
        stream, lir, diag = model_outputs(model, noisy, args.patch_size, args.lir_steps, args.lir_alpha)
        smooth = smooth_base_prediction(noisy, k=args.smooth_k, alpha=args.smooth_alpha)
        cd_noisy = chamfer_l2_np(noisy, clean)
        variants: list[tuple[str, np.ndarray]] = []
        for w in args.stream_weights:
            variants.append((f"w{w:g}", (w * stream + (1.0 - w) * lir).astype(np.float32)))
        for smooth_mix in args.smooth_mixes:
            if smooth_mix <= 0:
                continue
            for w in args.stream_weights:
                blend = (w * stream + (1.0 - w) * lir).astype(np.float32)
                variants.append((f"w{w:g}_smooth{smooth_mix:g}", ((1.0 - smooth_mix) * blend + smooth_mix * smooth).astype(np.float32)))
        variants.append(("smooth_only", smooth.astype(np.float32)))
        variants.append(("stream_only", stream.astype(np.float32)))
        variants.append(("lir_only", lir.astype(np.float32)))
        for variant, base in variants:
            cd_base = chamfer_l2_np(base, clean)
            move = np.sqrt(((base - noisy) ** 2).sum(axis=1))
            rows.append({
                "checkpoint": spec.name,
                "variant": variant,
                "sample_i": sample_i,
                "sample": s["sample"],
                "sigma": s["sigma"],
                "cd_noisy": cd_noisy,
                "cd_base": cd_base,
                "base_better_than_noisy": int(cd_base < cd_noisy),
                "score_vs_noisy": max(0.0, min(100.0, 100.0 * (1.0 - cd_base / cd_noisy))) if cd_noisy > 0 else 0.0,
                "move_l2_mean": float(move.mean()),
                "move_l2_p95": float(np.quantile(move, 0.95)),
                **diag,
            })
    return rows


def summarize(rows: list[dict[str, object]], target_rate: float) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for r in rows:
        groups.setdefault((str(r["checkpoint"]), str(r["variant"])), []).append(r)
    out = []
    for (checkpoint, variant), rs in sorted(groups.items()):
        cd_noisy = float(np.mean([float(r["cd_noisy"]) for r in rs]))
        cd_base = float(np.mean([float(r["cd_base"]) for r in rs]))
        rate = float(np.mean([int(r["base_better_than_noisy"]) for r in rs]))
        score = float(np.mean([float(r["score_vs_noisy"]) for r in rs]))
        move = float(np.mean([float(r["move_l2_mean"]) for r in rs]))
        out.append({
            "checkpoint": checkpoint,
            "variant": variant,
            "n": len(rs),
            "cd_noisy": cd_noisy,
            "cd_base": cd_base,
            "cd_gain": cd_noisy - cd_base,
            "base_better_than_noisy_rate": rate,
            "score_vs_noisy": score,
            "move_l2_mean": move,
            "pass_gate": int(cd_base < cd_noisy and rate >= target_rate),
        })
    out.sort(key=lambda r: (r["pass_gate"], r["base_better_than_noisy_rate"], -r["cd_base"]), reverse=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="dataset_train")
    p.add_argument("--eval-list", default="starter_code/datalist/validate.txt")
    p.add_argument("--out-dir", default="analysis/fixed075_proxy_base_scan_20260519")
    p.add_argument("--seed", type=int, default=20260519)
    p.add_argument("--eval-limit", type=int, default=48)
    p.add_argument("--num-points", type=int, default=512)
    p.add_argument("--noise-min", type=float, default=0.005)
    p.add_argument("--noise-max", type=float, default=0.02)
    p.add_argument("--patch-size", type=int, default=1000)
    p.add_argument("--lir-steps", type=int, default=2)
    p.add_argument("--lir-alpha", type=float, default=0.75)
    p.add_argument("--stream-weights", type=float, nargs="+", default=[0.6, 0.75, 0.9, 1.0])
    p.add_argument("--smooth-mixes", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    p.add_argument("--smooth-k", type=int, default=8)
    p.add_argument("--smooth-alpha", type=float, default=0.04)
    p.add_argument("--target-rate", type=float, default=0.75)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    jt.flags.use_cuda = 0 if args.cpu else 1
    out_dir = repo_path(args.out_dir); data_root = repo_path(args.data_root); eval_list = repo_path(args.eval_list)
    assert out_dir and data_root and eval_list
    out_dir.mkdir(parents=True, exist_ok=True)

    obj_files = list_obj_files(data_root, eval_list, args.eval_limit)
    rng = np.random.default_rng(args.seed + 9999)
    samples = []
    for i, (sample, obj) in enumerate(obj_files):
        clean, noisy, sigma = make_clean_noisy(obj, args.num_points, args.noise_min, args.noise_max, rng)
        samples.append({"sample": sample, "obj": str(obj), "clean": clean, "noisy": noisy, "sigma": sigma})

    all_rows = []
    for spec in candidate_specs():
        try:
            model = build_model(spec)
        except Exception as e:
            print(f"skip {spec.name}: {e}", flush=True)
            continue
        rows = eval_rows_for_spec(spec, model, samples, args)
        all_rows.extend(rows)
        print(f"scanned {spec.name}: {len(rows)} rows", flush=True)

    detail_csv = out_dir / "detail.csv"
    with detail_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    summary = summarize(all_rows, args.target_rate)
    summary_csv = out_dir / "summary.csv"
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)

    lines = ["# fixed075 proxy base health scan", "", f"target_rate: `{args.target_rate}`", "", "| checkpoint | variant | cd_noisy | cd_base | cd_gain | base_rate | score | pass |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary[:30]:
        lines.append(f"| {r['checkpoint']} | {r['variant']} | {r['cd_noisy']:.8g} | {r['cd_base']:.8g} | {r['cd_gain']:.8g} | {r['base_better_than_noisy_rate']:.3f} | {r['score_vs_noisy']:.3f} | {r['pass_gate']} |")
    passed = [r for r in summary if r["pass_gate"]]
    lines += ["", f"pass_count: {len(passed)}", "", "Decision: use a passed base for adapter diagnostics; if none pass, do not expand GARA-D training."]
    (out_dir / "diagnosis.md").write_text("\n".join(lines) + "\n")
    meta = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "args": vars(args), "summary_csv": str(summary_csv), "detail_csv": str(detail_csv), "pass_count": len(passed), "best": summary[0]}
    (out_dir / "summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
