#!/usr/bin/env python3
"""Calibrate KNN spacing / roughness refs for move-scale gates.

Outputs synthetic train low/mid/high distributions and official noisy test
statistics under one CSV/summary, without importing Jittor.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.spatial import cKDTree

from evaluate_noise_estimator import load_obj_vertices, normalize_pc, sample_points, list_obj_files, repo_path, knn_stats


def list_test_noisy(test_root: Path, limit: int) -> list[Path]:
    files = sorted(test_root.glob("shapenet/*/*/noisy.npy"))
    if limit > 0:
        files = files[:limit]
    if not files:
        raise RuntimeError(f"no noisy.npy files under {test_root}")
    return files


def summarize(rows: list[dict[str, object]], keys: list[str]) -> str:
    lines: list[str] = []
    groups = []
    for label in ["train_low", "train_mid", "train_high", "test_noisy"]:
        if any(r["group"] == label for r in rows):
            groups.append(label)
    for key in keys:
        lines.append(f"\n{key}")
        for g in groups:
            vs = np.asarray([float(r[key]) for r in rows if r["group"] == g], dtype=np.float64)
            if len(vs) == 0:
                continue
            lines.append(
                f"  {g:10s}: n={len(vs):4d} mean={vs.mean():.6f} "
                f"p05={np.quantile(vs,0.05):.6f} p10={np.quantile(vs,0.10):.6f} "
                f"med={np.median(vs):.6f} p90={np.quantile(vs,0.90):.6f} p95={np.quantile(vs,0.95):.6f}"
            )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="dataset_train")
    ap.add_argument("--test-root", default="dataset_test_noisy")
    ap.add_argument("--list", default="starter_code/datalist/train.txt")
    ap.add_argument("--train-limit", type=int, default=64)
    ap.add_argument("--test-limit", type=int, default=64)
    ap.add_argument("--num-points", type=int, default=2048)
    ap.add_argument("--test-sample-points", type=int, default=50000, help="0 means use all test points")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260512)
    ap.add_argument("--out", default="analysis/move_scale_calibration_20260512/stats.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    bands = {
        "train_low": (0.005, 0.008),
        "train_mid": (0.010, 0.014),
        "train_high": (0.018, 0.022),
    }

    train_files = list_obj_files(repo_path(args.data_root), repo_path(args.list), args.train_limit)
    for group, (nmin, nmax) in bands.items():
        for i, (sample, obj) in enumerate(train_files, 1):
            clean = sample_points(normalize_pc(load_obj_vertices(obj)), args.num_points, rng)
            sigma = float(rng.uniform(nmin, nmax))
            noisy = clean + rng.normal(0.0, sigma, clean.shape).astype(np.float32)
            stats = knn_stats(noisy, args.k)
            rows.append({"group": group, "idx": i, "sample": sample, "sigma": sigma, **stats})

    for i, path in enumerate(list_test_noisy(repo_path(args.test_root), args.test_limit), 1):
        pts = np.load(path).astype(np.float32)
        if args.test_sample_points and len(pts) > args.test_sample_points:
            pts = sample_points(pts, args.test_sample_points, rng)
        stats = knn_stats(pts, args.k)
        rel = path.relative_to(repo_path(args.test_root))
        rows.append({"group": "test_noisy", "idx": i, "sample": str(rel), "sigma": "", **stats})

    out = repo_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    keys = ["mean_knn_mean", "mean_knn_median", "mean_knn_p75", "kth_median", "plane_res_mean", "plane_res_median", "plane_res_p75"]
    summary = summarize(rows, keys)
    summary_path = out.with_suffix(".summary.txt")
    summary_path.write_text(f"out: {out}\ntrain_limit={args.train_limit} test_limit={args.test_limit} num_points={args.num_points} k={args.k}\n{summary}\n", encoding="utf-8")
    print(summary_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
