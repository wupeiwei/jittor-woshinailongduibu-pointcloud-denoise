#!/usr/bin/env python3
"""Inspect official VM patch coverage before denoising.

This is a CPU/lightweight structural diagnostic for the official baseline
stitching bug: FPS seed KNN patches are not guaranteed to cover every input
point, so uncovered points can be dropped by the original stitch loop.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def farthest_point_sampling_np(points: np.ndarray, num_patches: int) -> np.ndarray:
    # 纯 NumPy FPS，用来复现官方 patch seed 选择，不依赖 Jittor/CUDA。
    n = points.shape[0]
    selected: list[int] = []
    dist = np.full((n,), np.inf, dtype=np.float64)
    farthest = 0
    for _ in range(num_patches):
        selected.append(farthest)
        d = np.sum((points - points[farthest]) ** 2, axis=1)
        dist = np.minimum(dist, d)
        farthest = int(np.argmax(dist))
    return np.asarray(selected, dtype=np.int64)


def knn_indices_chunked(seeds: np.ndarray, points: np.ndarray, k: int, chunk: int) -> np.ndarray:
    # 分块算 KNN，避免一次构造过大的 seeds x points 距离矩阵。
    idxs = []
    for start in range(0, seeds.shape[0], chunk):
        seed_chunk = seeds[start:start + chunk]
        dist = np.sum((seed_chunk[:, None, :] - points[None, :, :]) ** 2, axis=-1)
        part = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
        idxs.append(part.astype(np.int64, copy=False))
    return np.concatenate(idxs, axis=0)


def inspect(path: Path, patch_size: int, seed_k: int, chunk: int) -> tuple[int, int, int]:
    # 只检查 patch 覆盖率，不跑模型；missing > 0 就说明原 stitch loop 有丢点风险。
    points = np.load(path, allow_pickle=False)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"not Nx3: {path} shape={points.shape}")
    n = points.shape[0]
    num_patches = int(seed_k * n / patch_size)
    seed_idx = farthest_point_sampling_np(points, num_patches)
    point_idxs = knn_indices_chunked(points[seed_idx], points, patch_size, chunk)
    covered = np.zeros((n,), dtype=np.bool_)
    covered[point_idxs.reshape(-1)] = True
    missing = int((~covered).sum())
    return n, num_patches, missing


def main() -> None:
    ap = argparse.ArgumentParser()
    # 传入一个或多个 noisy.npy/denoised.npy 都可以，本脚本只关心 Nx3 点集覆盖情况。
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--patch-size", type=int, default=1000)
    ap.add_argument("--seed-k", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=16)
    args = ap.parse_args()

    total_missing = 0
    for path in args.paths:
        n, num_patches, missing = inspect(path, args.patch_size, args.seed_k, args.chunk)
        total_missing += missing
        print(f"{path}: N={n} patches={num_patches} missing={missing}")
    print(f"total_missing={total_missing}")


if __name__ == "__main__":
    main()
