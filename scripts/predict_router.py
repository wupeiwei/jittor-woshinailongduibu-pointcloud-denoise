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
        staas_fusion=bool(model_cfg.get("staas_fusion", False)),
        staas_v2_gate=bool(model_cfg.get("staas_v2_gate", False)),
        staas_v2_geo_weight=float(model_cfg.get("staas_v2_geo_weight", 0.25)),
        staas_v2_gate_min=float(model_cfg.get("staas_v2_gate_min", 0.0)),
        staas_v2_gate_max=float(model_cfg.get("staas_v2_gate_max", 1.0)),
        staas_v2_noise_ref_low=float(model_cfg.get("staas_v2_noise_ref_low", 0.010)),
        staas_v2_noise_ref_high=float(model_cfg.get("staas_v2_noise_ref_high", 0.030)),
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


VETO_FEATURES = [
    "plane_res_mean",
    "plane_res_median",
    "plane_res_p75",
    "plane_res_p90",
    "plane_res_iqr",
    "edge_conf_mean",
    "edge_conf_p75",
    "scattering_mean",
    "scattering_p75",
    "mean_knn_p75",
    "scale_mean",
]

# Full-data ridge fit from analysis/veto_guard_probe512_20260520_summary.json
# Target: synthetic delta = ST-AAS-v2 score - baseline score. Positive means strong is better.
VETO_MU = np.array([
    0.010111018835573304,
    0.009473905752429346,
    0.011802821314972789,
    0.014539589176933976,
    0.004116646720461858,
    0.3115911893255543,
    0.4198972612066427,
    0.3021209001162788,
    0.3878704957169248,
    0.03662987137352842,
    0.033461914194049314,
], dtype=np.float64)
VETO_SD = np.array([
    0.004176866790226334,
    0.0040481358379597585,
    0.004944423791595409,
    0.005963308869620932,
    0.001892928823914533,
    0.0775502123536939,
    0.10690225084604998,
    0.057219387441940536,
    0.06297138341221802,
    0.013024378384181553,
    0.011428245787660917,
], dtype=np.float64)
VETO_COEF = np.array([
    14.747610792838934,
    2.2316584810845734,
    11.38634431336386,
    9.126683729480172,
    -1.773858552140475,
    -2.5578392436670074,
    4.722614168707231,
    -3.3567142722153425,
    -3.8524027094472033,
    -12.592531290738595,
    -14.1224349788964,
], dtype=np.float64)
VETO_INTERCEPT = -2.7975418878771507


def noisy_geometry_stats(points: np.ndarray, k: int = 16, max_points: int = 8192, seed: int = 1234) -> dict[str, float]:
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) > max_points:
        rng = np.random.default_rng(seed)
        pts = pts[rng.choice(len(pts), max_points, replace=False)]
    tree = cKDTree(pts)
    dists, idx = tree.query(pts, k=k + 1)
    dists = dists[:, 1:]
    idx = idx[:, 1:]
    neigh = pts[idx]
    rel = neigh - pts[:, None, :]
    local_dist = np.sqrt(np.maximum((rel * rel).sum(axis=-1), 0.0))
    mean_knn = dists.mean(axis=1)
    scale = np.sort(local_dist, axis=1)[:, k // 2]

    centered = neigh - neigh.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / max(k, 1)
    evals = np.linalg.eigvalsh(cov)
    l1 = np.maximum(evals[:, 2], 1e-12)
    l2 = np.maximum(evals[:, 1], 0.0)
    l3 = np.maximum(evals[:, 0], 0.0)
    linearity = np.clip((l1 - l2) / l1, 0.0, 1.0)
    scattering = np.clip(l3 / l1, 0.0, 1.0)
    edge_conf = np.clip(linearity * (1.0 - scattering), 0.0, 1.0)
    plane_res = np.sqrt(np.maximum(l3, 0.0))

    q25 = float(np.quantile(plane_res, 0.25))
    q75 = float(np.quantile(plane_res, 0.75))
    return {
        "plane_res_mean": float(plane_res.mean()),
        "plane_res_median": float(np.median(plane_res)),
        "plane_res_p75": q75,
        "plane_res_p90": float(np.quantile(plane_res, 0.90)),
        "plane_res_iqr": q75 - q25,
        "edge_conf_mean": float(edge_conf.mean()),
        "edge_conf_p75": float(np.quantile(edge_conf, 0.75)),
        "scattering_mean": float(scattering.mean()),
        "scattering_p75": float(np.quantile(scattering, 0.75)),
        "mean_knn_p75": float(np.quantile(mean_knn, 0.75)),
        "scale_mean": float(scale.mean()),
    }


def plane_res_p75(points: np.ndarray, k: int = 16, max_points: int = 8192, seed: int = 1234) -> float:
    return noisy_geometry_stats(points, k=k, max_points=max_points, seed=seed)["plane_res_p75"]


def veto_pred_delta(stats: dict[str, float]) -> float:
    x = np.array([stats[k] for k in VETO_FEATURES], dtype=np.float64)
    z = (x - VETO_MU) / np.maximum(VETO_SD, 1e-12)
    return float(VETO_INTERCEPT + z @ VETO_COEF)


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
    p.add_argument("--router-mode", choices=["threshold", "plane_veto"], default="threshold")
    p.add_argument("--veto-margin", type=float, default=-2.0, help="plane_veto requires predicted_delta above this")
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
    fields = [
        "idx",
        "input",
        "output",
        "num_points",
        *VETO_FEATURES,
        "veto_pred_delta",
        "route",
        "elapsed_sec",
    ]
    low_count = strong_count = 0
    t_all = time.time()
    with route_csv.open("w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fields)
        writer.writeheader()
        for i, f in enumerate(files, 1):
            t0 = time.time()
            noisy_np = np.load(f).astype(np.float32)
            stats = noisy_geometry_stats(noisy_np, k=args.estimator_k, max_points=args.estimator_max_points, seed=args.seed + i)
            score = stats["plane_res_p75"]
            pred_delta = veto_pred_delta(stats) if args.router_mode == "plane_veto" else float("nan")
            if args.router_mode == "plane_veto":
                use_strong = score >= args.low_threshold and pred_delta > args.veto_margin
            else:
                use_strong = score >= args.low_threshold
            if not use_strong:
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
            row = {
                "idx": i,
                "input": str(rel),
                "output": str(out.relative_to(out_dir)),
                "num_points": len(noisy_np),
                "veto_pred_delta": pred_delta,
                "route": route,
                "elapsed_sec": elapsed,
            }
            row.update(stats)
            writer.writerow(row)
            fcsv.flush()
            print(
                f"[{i}/{len(files)}] route={route:6s} plane_res_p75={score:.6f} "
                f"veto_delta={pred_delta:.3f} shape={pred.shape} out={out}",
                flush=True,
            )

    print("summary")
    print(f"files: {len(files)} low={low_count} strong={strong_count} threshold={args.low_threshold:.6f} mode={args.router_mode} veto_margin={args.veto_margin:.3f}")
    print(f"routes: {route_csv}")
    print(f"elapsed_sec: {time.time() - t_all:.1f}")
    if not args.no_zip:
        make_zip(out_dir, zip_path)


if __name__ == "__main__":
    main()
