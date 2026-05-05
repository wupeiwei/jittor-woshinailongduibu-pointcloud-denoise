#!/usr/bin/env python3
"""Evaluate test-time noise/roughness estimators on synthetic noisy point clouds.

No Jittor import on purpose. This script estimates noise level from noisy points
only, then checks whether low/mid/high bands are separable enough for checkpoint
routing.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]


def repo_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load_obj_vertices(path: Path) -> np.ndarray:
    pts = []
    with path.open("r", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not pts:
        raise RuntimeError(f"no vertices in {path}")
    return np.asarray(pts, dtype=np.float32)


def normalize_pc(pc: np.ndarray) -> np.ndarray:
    pc = pc.astype(np.float32)
    pc = pc - pc.mean(axis=0, keepdims=True)
    scale = np.sqrt((pc * pc).sum(axis=1)).max()
    if scale > 1e-12:
        pc = pc / scale
    return pc.astype(np.float32)


def list_obj_files(data_root: Path, list_file: Path, limit: int) -> list[tuple[str, Path]]:
    ids = [x.strip() for x in list_file.read_text().splitlines() if x.strip()]
    files = []
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


def sample_points(pc: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.choice(len(pc), n, replace=len(pc) < n)
    return pc[idx].astype(np.float32)


def knn_stats(noisy: np.ndarray, k: int) -> dict[str, float]:
    tree = cKDTree(noisy)
    d, idx = tree.query(noisy, k=k + 1)
    d = d[:, 1:]
    idx = idx[:, 1:]
    mean_knn = d.mean(axis=1)
    kth = d[:, -1]

    neigh = noisy[idx]
    centered = neigh - neigh.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / max(k, 1)
    evals = np.linalg.eigvalsh(cov)
    plane_res = np.sqrt(np.maximum(evals[:, 0], 0.0))

    return {
        "mean_knn_mean": float(mean_knn.mean()),
        "mean_knn_median": float(np.median(mean_knn)),
        "mean_knn_p75": float(np.quantile(mean_knn, 0.75)),
        "kth_median": float(np.median(kth)),
        "plane_res_mean": float(plane_res.mean()),
        "plane_res_median": float(np.median(plane_res)),
        "plane_res_p75": float(np.quantile(plane_res, 0.75)),
    }


def best_threshold(vals: list[tuple[float, str]], positive: set[str]) -> tuple[float, float]:
    xs = sorted(set(v for v, _ in vals))
    if len(xs) <= 1:
        return (xs[0] if xs else 0.0), 0.0
    cands = [(xs[i] + xs[i + 1]) / 2 for i in range(len(xs) - 1)]
    best_t, best_acc = cands[0], -1.0
    for t in cands:
        ok = 0
        for v, label in vals:
            pred_pos = v >= t
            truth = label in positive
            ok += int(pred_pos == truth)
        acc = ok / len(vals)
        if acc > best_acc:
            best_t, best_acc = t, acc
    return best_t, best_acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="dataset_train")
    ap.add_argument("--list", default="starter_code/datalist/train.txt")
    ap.add_argument("--limit", type=int, default=128)
    ap.add_argument("--num-points", type=int, default=2048)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="experiments/evals/noise_estimator_128.csv")
    args = ap.parse_args()

    bands = {"low": (0.005, 0.008), "mid": (0.010, 0.014), "high": (0.018, 0.022)}
    rng = np.random.default_rng(args.seed)
    files = list_obj_files(repo_path(args.data_root), repo_path(args.list), args.limit)
    out = repo_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for band, (nmin, nmax) in bands.items():
        for i, (sample, obj) in enumerate(files, 1):
            clean = sample_points(normalize_pc(load_obj_vertices(obj)), args.num_points, rng)
            sigma = float(rng.uniform(nmin, nmax))
            noisy = clean + rng.normal(0.0, sigma, clean.shape).astype(np.float32)
            stats = knn_stats(noisy, args.k)
            rows.append({"band": band, "idx": i, "sample": sample, "sigma": sigma, **stats})

    fields = list(rows[0].keys())
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    print(f"out: {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    for key in ["mean_knn_mean", "mean_knn_median", "kth_median", "plane_res_mean", "plane_res_median", "plane_res_p75"]:
        print(f"\n{key}")
        for band in ["low", "mid", "high"]:
            vs = [float(r[key]) for r in rows if r["band"] == band]
            print(f"  {band:4s}: mean={mean(vs):.6f} p10={np.quantile(vs,0.10):.6f} med={np.median(vs):.6f} p90={np.quantile(vs,0.90):.6f}")
        vals = [(float(r[key]), r["band"]) for r in rows]
        t_low, acc_low = best_threshold(vals, {"mid", "high"})
        t_high, acc_high = best_threshold(vals, {"high"})
        print(f"  best low-vs-rest threshold:  {t_low:.6f}, acc={acc_low:.3f}")
        print(f"  best high-vs-rest threshold: {t_high:.6f}, acc={acc_high:.3f}")


if __name__ == "__main__":
    main()
