#!/usr/bin/env python3
"""Probe noisy-only adaptive blend weights for hidden-test submissions.

This script does not train or submit. It reads two submission zips with matching
entries, computes a cheap noisy-only ST-AAS-style geometry confidence from the
hidden noisy cloud, and writes adaptive blend submission zips plus diagnostics.

Formula:
    denoised = w_stream * stream + (1 - w_stream) * lir

The default rules are deliberately conservative around the known best fixed
blend (stream=0.75, lir=0.25). They only move the LIR fraction within a small
range based on noisy-only confidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
# diagnostics 字段用于解释 blend 权重来源和位移幅度，不代表官方分数。
FIELDS = [
    "entry",
    "category",
    "model_id",
    "rule",
    "n_points",
    "plane_res_p75",
    "edge_conf_mean",
    "edge_conf_p75",
    "scattering_mean",
    "stream_weight",
    "lir_weight",
    "disp_from_stream_mean",
    "disp_from_stream_p95",
    "disp_from_lir_mean",
    "disp_from_lir_p95",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def npy_from_zip(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    with zf.open(name, "r") as f:
        return np.load(io.BytesIO(f.read()), allow_pickle=False)


def write_npy_to_zip(zf: zipfile.ZipFile, name: str, arr: np.ndarray) -> None:
    buf = io.BytesIO()
    np.save(buf, arr.astype(np.float32, copy=False), allow_pickle=False)
    zf.writestr(name, buf.getvalue())


def noisy_path_from_entry(test_root: Path, entry: str) -> Path:
    rel = Path(entry)
    return test_root / rel.parent / "noisy.npy"


def sample_for_stats(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) <= max_points:
        return pts
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts), max_points, replace=False)
    return pts[idx]


def geometry_stats(points: np.ndarray, k: int = 16, max_points: int = 8192, seed: int = 1234) -> dict[str, float]:
    # ST-AAS 风格的轻量几何统计：平面残差、边缘置信和散乱度都只从 noisy 计算。
    pts = sample_for_stats(points, max_points=max_points, seed=seed)
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=k + 1)
    neigh = pts[idx[:, 1:]]
    center = pts[:, None, :]
    rel = neigh - center
    dist = np.sqrt(np.maximum((rel * rel).sum(axis=-1), 0.0))
    scale = np.sort(dist, axis=1)[:, k // 2]

    centered = neigh - neigh.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / float(k)
    evals = np.linalg.eigvalsh(cov)
    l1 = np.maximum(evals[:, 2], 1e-12)
    l2 = np.maximum(evals[:, 1], 0.0)
    l3 = np.maximum(evals[:, 0], 0.0)
    linearity = np.clip((l1 - l2) / l1, 0.0, 1.0)
    scattering = np.clip(l3 / l1, 0.0, 1.0)
    edge_conf = np.clip(linearity * (1.0 - scattering), 0.0, 1.0)
    plane_res = np.sqrt(np.maximum(l3, 0.0))
    return {
        "plane_res_p75": float(np.quantile(plane_res, 0.75)),
        "edge_conf_mean": float(edge_conf.mean()),
        "edge_conf_p75": float(np.quantile(edge_conf, 0.75)),
        "scattering_mean": float(scattering.mean()),
        "scale_mean": float(scale.mean()),
    }


def stream_weight(rule: str, s: dict[str, float]) -> float:
    # Fixed best reference: stream 0.75, LIR 0.25.
    # 这些规则都是围绕 fixed075 的小范围扰动；不要把它们直接当最终推荐候选。
    if rule == "fixed075":
        return 0.75
    if rule == "edge_guard":
        # More edge-like / low-scatter clouds are safer to keep close to stream;
        # scattered/noisy-looking clouds get slightly more LIR. Conservative range.
        edge = s["edge_conf_p75"]
        scatter = s["scattering_mean"]
        w = 0.75 + 0.12 * (edge - 0.50) - 0.08 * (scatter - 0.20)
        return float(np.clip(w, 0.68, 0.84))
    if rule == "noise_gate":
        # Plane residual is the old noisy-only signal. Around hidden-test stats,
        # move only mildly away from fixed blend.
        r = s["plane_res_p75"]
        w = 0.84 - 10.0 * (r - 0.010)
        return float(np.clip(w, 0.66, 0.86))
    if rule == "hybrid_guard":
        edge = s["edge_conf_p75"]
        scatter = s["scattering_mean"]
        r = s["plane_res_p75"]
        w = 0.77 + 0.10 * (edge - 0.50) - 0.06 * (scatter - 0.20) - 6.0 * (r - 0.010)
        return float(np.clip(w, 0.68, 0.84))
    raise ValueError(f"unknown rule: {rule}")


def main() -> None:
    p = argparse.ArgumentParser()
    # 默认路径是历史本机 artifact；在新环境运行时应显式传 stream/lir zip。
    p.add_argument("--stream-zip", required=True, help="stream/VM submission zip; pass explicitly, no machine-local default")
    p.add_argument("--lir-zip", required=True, help="LIR submission zip; pass explicitly, no machine-local default")
    p.add_argument("--test-root", default="dataset_test_noisy")
    p.add_argument("--out-dir", default="analysis/adaptive_blend_probe_20260517")
    p.add_argument("--rules", nargs="+", default=["fixed075", "edge_guard", "noise_gate", "hybrid_guard"])
    p.add_argument("--max-stat-points", type=int, default=8192)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    stream_zip = Path(args.stream_zip)
    lir_zip = Path(args.lir_zip)
    test_root = Path(args.test_root)
    if not test_root.is_absolute():
        test_root = ROOT / test_root
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "stream_zip": str(stream_zip),
        "stream_sha256": sha256(stream_zip),
        "lir_zip": str(lir_zip),
        "lir_sha256": sha256(lir_zip),
        "test_root": str(test_root),
        "rules": {},
    }

    with zipfile.ZipFile(stream_zip, "r") as zs, zipfile.ZipFile(lir_zip, "r") as zl:
        names_s = sorted(n for n in zs.namelist() if n.endswith("denoised.npy"))
        names_l = sorted(n for n in zl.namelist() if n.endswith("denoised.npy"))
        if names_s != names_l:
            raise RuntimeError("stream/lir zip entries do not match")

        stats_cache: dict[str, dict[str, float]] = {}
        for i, name in enumerate(names_s):
            # 几何统计对每个条目缓存一次，多个 blend rule 复用同一 noisy-only 特征。
            noisy_path = noisy_path_from_entry(test_root, name)
            if not noisy_path.exists():
                raise FileNotFoundError(noisy_path)
            noisy = np.load(noisy_path, allow_pickle=False)
            stats_cache[name] = geometry_stats(noisy, k=args.k, max_points=args.max_stat_points, seed=args.seed + i)

        for rule in args.rules:
            # 每个 rule 输出一份候选 zip 和 diagnostics，便于后续 check_submission/candidate_suite 审查。
            zip_path = out_dir / f"result_adaptive_blend_{rule}_20260517.zip"
            csv_path = out_dir / f"diagnostics_{rule}.csv"
            weights = []
            disp_stream_means = []
            disp_lir_means = []
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zout, csv_path.open("w", newline="") as fcsv:
                writer = csv.DictWriter(fcsv, fieldnames=FIELDS)
                writer.writeheader()
                for name in names_s:
                    stream = npy_from_zip(zs, name).astype(np.float32, copy=False)
                    lir = npy_from_zip(zl, name).astype(np.float32, copy=False)
                    s = stats_cache[name]
                    w = stream_weight(rule, s)
                    out = (w * stream + (1.0 - w) * lir).astype(np.float32)
                    write_npy_to_zip(zout, name, out)
                    ds = np.sqrt(((out - stream) ** 2).sum(axis=-1))
                    dl = np.sqrt(((out - lir) ** 2).sum(axis=-1))
                    parts = Path(name).parts
                    row = {
                        "entry": name,
                        "category": parts[1] if len(parts) >= 4 else "",
                        "model_id": parts[2] if len(parts) >= 4 else "",
                        "rule": rule,
                        "n_points": int(out.shape[0]),
                        **s,
                        "stream_weight": w,
                        "lir_weight": 1.0 - w,
                        "disp_from_stream_mean": float(ds.mean()),
                        "disp_from_stream_p95": float(np.quantile(ds, 0.95)),
                        "disp_from_lir_mean": float(dl.mean()),
                        "disp_from_lir_p95": float(np.quantile(dl, 0.95)),
                    }
                    writer.writerow({k: row[k] for k in FIELDS})
                    weights.append(w)
                    disp_stream_means.append(row["disp_from_stream_mean"])
                    disp_lir_means.append(row["disp_from_lir_mean"])

            summary["rules"][rule] = {
                "zip": str(zip_path),
                "sha256": sha256(zip_path),
                "diagnostics_csv": str(csv_path),
                "files": len(names_s),
                "stream_weight_mean": float(np.mean(weights)),
                "stream_weight_min": float(np.min(weights)),
                "stream_weight_max": float(np.max(weights)),
                "lir_weight_mean": float(1.0 - np.mean(weights)),
                "disp_from_stream_mean": float(np.mean(disp_stream_means)),
                "disp_from_lir_mean": float(np.mean(disp_lir_means)),
            }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
