#!/usr/bin/env python3
"""Inference-time checkpoint router for point cloud denoising.

Routes each noisy test cloud to a conservative low-noise checkpoint or a strong
mid/high-noise checkpoint using a noisy-only local PCA roughness estimator.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jittor as jt

from denoise_baseline import ResidualDenoiser, deep_update, predict_points_in_chunks


def repo_path(path: str | Path | None) -> Path | None:
    if path is None or str(path) == "":
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def load_config(config: Path, profiles: list[Path]) -> dict[str, Any]:
    cfg = read_yaml(config)
    for profile in profiles:
        cfg = deep_update(cfg, read_yaml(profile))
    return cfg


def build_model(cfg: dict[str, Any], ckpt: Path) -> ResidualDenoiser:
    model_cfg = cfg.get("model", {})
    model = ResidualDenoiser(
        k=int(model_cfg.get("k", 16)),
        feat_dim=int(model_cfg.get("feat_dim", 256)),
        hidden=int(model_cfg.get("hidden", 256)),
        use_pwsenel=bool(model_cfg.get("pwsenel", False)),
        use_staas=bool(model_cfg.get("staas", False)),
        staas_strength=float(model_cfg.get("staas_strength", 1.0)),
        staas_tau0=float(model_cfg.get("staas_tau0", 0.02)),
        staas_tau_min=float(model_cfg.get("staas_tau_min", 0.005)),
        staas_tau_max=float(model_cfg.get("staas_tau_max", 0.08)),
        use_move_gate=bool(model_cfg.get("move_gate", False)),
        use_pwsenel_v2=bool(model_cfg.get("pwsenel_v2", False)),
        pwsenel_v2_edge_lock=float(model_cfg.get("pwsenel_v2_edge_lock", 0.7)),
        pwsenel_v2_gate_scale=float(model_cfg.get("pwsenel_v2_gate_scale", 0.5)),
        residual_clip=float(model_cfg.get("residual_clip", 0.0)),
        adaptive_clip=bool(model_cfg.get("adaptive_clip", False)),
        adaptive_clip_min=float(model_cfg.get("adaptive_clip_min", 0.006)),
        adaptive_clip_max=float(model_cfg.get("adaptive_clip_max", 0.020)),
        adaptive_clip_ref_low=float(model_cfg.get("adaptive_clip_ref_low", 0.022)),
        adaptive_clip_ref_mid=float(model_cfg.get("adaptive_clip_ref_mid", 0.030)),
        adaptive_clip_ref_high=float(model_cfg.get("adaptive_clip_ref_high", 0.040)),
        adaptive_clip_mid=float(model_cfg.get("adaptive_clip_mid", 0.010)),
        noise_aware_move_gate=bool(model_cfg.get("noise_aware_move_gate", False)),
        noise_aware_gate_min=float(model_cfg.get("noise_aware_gate_min", 0.45)),
        noise_aware_gate_ref_low=float(model_cfg.get("noise_aware_gate_ref_low", 0.022)),
        noise_aware_gate_ref_high=float(model_cfg.get("noise_aware_gate_ref_high", 0.036)),
        hybrid_safe_strong=bool(model_cfg.get("hybrid_safe_strong", False)),
        hybrid_router_scale=float(model_cfg.get("hybrid_router_scale", 1.0)),
    )
    model.load(str(ckpt))
    model.eval()
    return model


def plane_res_p75(points: np.ndarray, k: int = 16, max_points: int = 8192, seed: int = 1234) -> float:
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) > max_points:
        rng = np.random.default_rng(seed)
        pts = pts[rng.choice(len(pts), max_points, replace=False)]
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=k + 1)
    idx = idx[:, 1:]
    neigh = pts[idx]
    centered = neigh - neigh.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / max(k, 1)
    evals = np.linalg.eigvalsh(cov)
    residual = np.sqrt(np.maximum(evals[:, 0], 0.0))
    return float(np.quantile(residual, 0.75))


def make_zip(out_dir: Path, zip_path: Path) -> None:
    files = sorted(out_dir.glob("shapenet/*/*/denoised.npy"))
    if not files:
        raise RuntimeError(f"no denoised.npy files under {out_dir}")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, f.relative_to(out_dir))
    print(f"wrote {zip_path} files={len(files)}")


def main() -> None:
    p = argparse.ArgumentParser(description="Predict with noisy-only low/mid-high checkpoint router.")
    p.add_argument("--low-config", default="configs/denoise_pwsenel_v2_adaptive_clip_piecewise.yaml")
    p.add_argument("--strong-config", default="configs/denoise_noise_aware_move_gate.yaml")
    p.add_argument("--profile", action="append", default=[])
    p.add_argument("--low-ckpt", default="experiments/denoise_pwsenel_v2_adaptive_clip_piecewise/pwsenel_v2_adaptive_clip_piecewise.pkl")
    p.add_argument("--strong-ckpt", default="experiments/denoise_noise_aware_move_gate/noise_aware_move_gate.pkl")
    p.add_argument("--test-root", default="dataset_test_noisy")
    p.add_argument("--out-dir", default="results/denoise_router")
    p.add_argument("--zip", default="result_denoise_router.zip")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--patch-size", type=int, default=8192)
    p.add_argument("--estimator-k", type=int, default=16)
    p.add_argument("--estimator-max-points", type=int, default=8192)
    p.add_argument("--low-threshold", type=float, default=0.006659, help="plane_res_p75 below this uses low checkpoint")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-zip", action="store_true")
    args = p.parse_args()

    jt.flags.use_cuda = 0 if args.cpu else 1
    profiles = [repo_path(x) for x in args.profile]
    low_cfg = load_config(repo_path(args.low_config), profiles)
    strong_cfg = load_config(repo_path(args.strong_config), profiles)
    low_ckpt = repo_path(args.low_ckpt)
    strong_ckpt = repo_path(args.strong_ckpt)
    test_root = repo_path(args.test_root)
    out_dir = repo_path(args.out_dir)
    zip_path = repo_path(args.zip)
    assert low_ckpt and strong_ckpt and test_root and out_dir and zip_path
    for pth in [low_ckpt, strong_ckpt, test_root]:
        if not pth.exists():
            raise FileNotFoundError(str(pth))

    print(f"loading low:    {low_ckpt}")
    low_model = build_model(low_cfg, low_ckpt)
    print(f"loading strong: {strong_ckpt}")
    strong_model = build_model(strong_cfg, strong_ckpt)

    files = sorted(test_root.glob("shapenet/*/*/noisy.npy"))
    if args.limit > 0:
        files = files[:args.limit]
    if not files:
        raise RuntimeError(f"no noisy.npy files under {test_root}")
    out_dir.mkdir(parents=True, exist_ok=True)
    route_csv = out_dir / "router_routes.csv"
    fields = ["idx", "input", "output", "num_points", "plane_res_p75", "route", "elapsed_sec"]
    low_count = strong_count = 0
    t_all = time.time()
    with route_csv.open("w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fields)
        writer.writeheader()
        for i, f in enumerate(files, 1):
            t0 = time.time()
            noisy_np = np.load(f).astype(np.float32)
            score = plane_res_p75(noisy_np, k=args.estimator_k, max_points=args.estimator_max_points, seed=args.seed + i)
            if score < args.low_threshold:
                route = "low"
                model = low_model
                low_count += 1
            else:
                route = "strong"
                model = strong_model
                strong_count += 1
            pred = predict_points_in_chunks(model, noisy_np, args.patch_size)
            rel = f.relative_to(test_root)
            out = out_dir / rel.parent / "denoised.npy"
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out, pred.astype(np.float32))
            elapsed = time.time() - t0
            writer.writerow({
                "idx": i,
                "input": str(rel),
                "output": str(out.relative_to(out_dir)),
                "num_points": len(noisy_np),
                "plane_res_p75": score,
                "route": route,
                "elapsed_sec": elapsed,
            })
            fcsv.flush()
            print(f"[{i}/{len(files)}] route={route:6s} plane_res_p75={score:.6f} shape={pred.shape} out={out}", flush=True)

    print("summary")
    print(f"files: {len(files)} low={low_count} strong={strong_count} threshold={args.low_threshold:.6f}")
    print(f"routes: {route_csv}")
    print(f"elapsed_sec: {time.time() - t_all:.1f}")
    if not args.no_zip:
        make_zip(out_dir, zip_path)


if __name__ == "__main__":
    main()
