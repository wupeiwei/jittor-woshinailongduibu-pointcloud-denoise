#!/usr/bin/env python3
"""Lightweight synthetic CD evaluation for denoising experiments.

This evaluates a saved checkpoint on clean training OBJ models with freshly
synthesized Gaussian noise, then compares:
- CD(noisy, clean)
- CD(pred, clean)
- score = clamp(100 * (1 - CD_pred / CD_noisy), 0, 100)

It is not the official A/B leaderboard metric because official hidden clean
surfaces are unavailable locally. It is meant for fast ablation sanity checks:
if a model cannot beat synthetic noisy input here, it is very unlikely to score
well on the leaderboard.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jittor as jt

from denoise_baseline import ResidualDenoiser, deep_update, load_obj_vertices, normalize_pc


FIELDS = [
    "idx",
    "sample",
    "category",
    "model_id",
    "sigma",
    "num_points",
    "cd_noisy",
    "cd_pred",
    "cd_ratio",
    "cd_score",
    "pred_better",
    "pred_offset_abs_mean",
    "pred_offset_l2_mean",
    "elapsed_sec",
]


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


def list_obj_files(data_root: Path, list_file: Path, limit: int) -> list[tuple[str, Path]]:
    ids = [x.strip() for x in list_file.read_text().splitlines() if x.strip()]
    files: list[tuple[str, Path]] = []
    for sample in ids:
        rel = Path(sample)
        if rel.parts and rel.parts[0] == "shapenet":
            obj = data_root / rel / "models" / "model_normalized.obj"
        else:
            obj = data_root / "shapenet" / rel / "models" / "model_normalized.obj"
        if obj.exists():
            files.append((sample, obj))
        if limit > 0 and len(files) >= limit:
            break
    if not files:
        raise RuntimeError(f"no obj files found from {list_file} under {data_root}")
    return files


def sample_points_rng(pc: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    replace = len(pc) < n
    idx = rng.choice(len(pc), n, replace=replace)
    return pc[idx].astype(np.float32)


def chamfer_l2_np(a: np.ndarray, b: np.ndarray) -> float:
    tree_b = cKDTree(b)
    dist_a2b, _ = tree_b.query(a, k=1)
    tree_a = cKDTree(a)
    dist_b2a, _ = tree_a.query(b, k=1)
    return float((dist_a2b ** 2).mean() + (dist_b2a ** 2).mean())


def metric_to_score(cd_pred: float, cd_noisy: float) -> float:
    if cd_noisy < 1e-15:
        return 100.0 if cd_pred < 1e-15 else 0.0
    return max(0.0, min(100.0, 100.0 * (1.0 - cd_pred / cd_noisy)))


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


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate checkpoint CD on synthetic noisy train/val OBJ samples.")
    p.add_argument("--config", default="configs/denoise_baseline.yaml")
    p.add_argument("--profile", action="append", default=[])
    p.add_argument("--ckpt", default="", help="Override checkpoint path")
    p.add_argument("--data-root", default="", help="Override clean OBJ root")
    p.add_argument("--list", default="", help="Override datalist file")
    p.add_argument("--out", default="experiments/eval_cd.csv")
    p.add_argument("--limit", type=int, default=32)
    p.add_argument("--num-points", type=int, default=0)
    p.add_argument("--noise-min", type=float, default=-1.0)
    p.add_argument("--noise-max", type=float, default=-1.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--cpu", action="store_true", help="Use CPU instead of CUDA")
    args = p.parse_args()

    cfg = load_config(repo_path(args.config), [repo_path(x) for x in args.profile])
    paths = cfg.get("paths", {})
    train = cfg.get("train", {})

    data_root = repo_path(args.data_root or paths.get("data_root", "dataset_train"))
    list_file = repo_path(args.list or paths.get("train_list", "starter_code/datalist/train.txt"))
    ckpt = repo_path(args.ckpt or paths.get("ckpt", "experiments/denoise_baseline/baseline.pkl"))
    out = repo_path(args.out)
    assert data_root is not None and list_file is not None and ckpt is not None and out is not None
    if not ckpt.exists():
        raise FileNotFoundError(f"ckpt not found: {ckpt}")

    num_points = args.num_points or int(train.get("num_points", 2048))
    noise_min = float(train.get("noise_min", 0.005)) if args.noise_min < 0 else args.noise_min
    noise_max = float(train.get("noise_max", 0.02)) if args.noise_max < 0 else args.noise_max

    jt.flags.use_cuda = 0 if args.cpu else 1
    rng = np.random.default_rng(args.seed)
    model = build_model(cfg, ckpt)
    samples = list_obj_files(data_root, list_file, args.limit)

    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    t_all = time.time()
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for i, (sample, obj_path) in enumerate(samples, 1):
            t0 = time.time()
            clean_full = normalize_pc(load_obj_vertices(obj_path))
            clean = sample_points_rng(clean_full, num_points, rng)
            sigma = float(rng.uniform(noise_min, noise_max))
            noisy = clean + rng.normal(0.0, sigma, clean.shape).astype(np.float32)

            with jt.no_grad():
                pred, offset = model(jt.array(noisy[None, ...]), return_offset=True)
                pred_np = np.asarray(pred.numpy()[0], dtype=np.float32)
                offset_np = np.asarray(offset.numpy()[0], dtype=np.float32)

            cd_noisy = chamfer_l2_np(noisy, clean)
            cd_pred = chamfer_l2_np(pred_np, clean)
            cd_ratio = cd_pred / cd_noisy if cd_noisy > 1e-15 else float("inf")
            cd_score = metric_to_score(cd_pred, cd_noisy)
            parts = Path(sample).parts
            row = {
                "idx": i,
                "sample": sample,
                "category": parts[1] if len(parts) >= 3 and parts[0] == "shapenet" else "",
                "model_id": parts[2] if len(parts) >= 3 and parts[0] == "shapenet" else "",
                "sigma": sigma,
                "num_points": num_points,
                "cd_noisy": cd_noisy,
                "cd_pred": cd_pred,
                "cd_ratio": cd_ratio,
                "cd_score": cd_score,
                "pred_better": int(cd_pred < cd_noisy),
                "pred_offset_abs_mean": float(np.abs(offset_np).mean()),
                "pred_offset_l2_mean": float(np.sqrt((offset_np ** 2).sum(axis=-1) + 1e-12).mean()),
                "elapsed_sec": time.time() - t0,
            }
            rows.append(row)
            writer.writerow(row)
            f.flush()
            print(
                f"[{i}/{len(samples)}] score={cd_score:.2f} ratio={cd_ratio:.3f} "
                f"cd_noisy={cd_noisy:.6g} cd_pred={cd_pred:.6g} sample={sample}",
                flush=True,
            )

    scores = np.array([float(r["cd_score"]) for r in rows], dtype=np.float64)
    ratios = np.array([float(r["cd_ratio"]) for r in rows], dtype=np.float64)
    better = np.array([int(r["pred_better"]) for r in rows], dtype=np.float64)
    print("\nsummary")
    print(f"out: {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    print(f"samples: {len(rows)}")
    print(f"mean_cd_score: {scores.mean():.4f}")
    print(f"median_cd_score: {np.median(scores):.4f}")
    print(f"mean_cd_ratio: {ratios.mean():.4f}")
    print(f"pred_better_rate: {better.mean() * 100.0:.1f}%")
    print(f"elapsed_sec: {time.time() - t_all:.1f}")


if __name__ == "__main__":
    main()
