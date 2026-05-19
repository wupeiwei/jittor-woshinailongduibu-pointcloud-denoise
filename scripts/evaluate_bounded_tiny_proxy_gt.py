#!/usr/bin/env python3
"""Evaluate bounded tiny adapter directions on train-split fixed075 proxy cache.

Uses cached clean/noisy/base samples from train_garad_fixed075_proxy.py outputs.
No hidden-test GT; no training. This tests whether the artifact-level shrink vs
extend intuition also holds against clean CD on a local synthetic split.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]


def chamfer_l2_np(a: np.ndarray, b: np.ndarray) -> float:
    ta = cKDTree(a)
    tb = cKDTree(b)
    da, _ = tb.query(a, k=1)
    db, _ = ta.query(b, k=1)
    return float((da ** 2).mean() + (db ** 2).mean())


def metric_to_score(cd_pred: float, cd_base: float) -> float:
    if cd_base <= 1e-15:
        return 100.0 if cd_pred <= cd_base else 0.0
    return max(0.0, min(100.0, 100.0 * (1.0 - cd_pred / cd_base)))


def repo_path(path: str | Path | None) -> Path | None:
    if path is None or str(path) == "":
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def sample_for_stats(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) <= max_points:
        return pts
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts), max_points, replace=False)
    return pts[idx]


def geometry_stats(points: np.ndarray, *, k: int, max_points: int, seed: int) -> dict[str, float]:
    pts = sample_for_stats(points, max_points=max_points, seed=seed)
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=k + 1)
    neigh = pts[idx[:, 1:]]
    rel = neigh - pts[:, None, :]
    dist = np.sqrt(np.maximum((rel * rel).sum(axis=-1), 0.0))
    scale = np.sort(dist, axis=1)[:, max(0, k // 2 - 1)]
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
        "edge_conf_p75": float(np.quantile(edge_conf, 0.75)),
        "scale_mean": float(scale.mean()),
    }


def rank01(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    out = np.empty_like(arr)
    if len(arr) == 1:
        out[0] = 0.5
    else:
        out[order] = np.linspace(0.0, 1.0, len(arr))
    return out


def finite_stats(xs: list[float]) -> dict[str, float]:
    arr = np.asarray(xs, dtype=np.float64)
    return {"mean": float(arr.mean()), "p50": float(np.quantile(arr, .5)), "p95": float(np.quantile(arr, .95)), "max": float(arr.max())}


def apply_delta(noisy: np.ndarray, base: np.ndarray, *, alpha: float, max_step_ratio: float, sign: float, gate: float, scale_mean: float) -> tuple[np.ndarray, dict[str, float]]:
    direction = base - noisy
    direction_l2 = np.sqrt(np.maximum((direction * direction).sum(axis=-1, keepdims=True), 1e-18))
    unit = direction / direction_l2
    raw_step = alpha * gate * direction_l2[..., 0]
    max_step = max_step_ratio * max(float(scale_mean), 1e-8)
    step = np.minimum(raw_step, max_step)
    delta = (sign * step[:, None] * unit).astype(np.float32)
    pred = (base + delta).astype(np.float32)
    dl2 = np.sqrt(np.maximum((delta * delta).sum(axis=-1), 0.0))
    return pred, {"delta_l2_mean": float(dl2.mean()), "delta_l2_p95": float(np.quantile(dl2, .95)), "delta_l2_max": float(dl2.max()), "clipped_frac": float((raw_step > max_step).mean())}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default="analysis/garad_v01_fixed075_proxy_confirm_20260519/garad_cd1p0_seed20260519/cache/eval")
    p.add_argument("--out-dir", default="analysis/garad_bounded_tiny_proxy_eval_20260519")
    p.add_argument("--limit", type=int, default=48)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--max-stat-points", type=int, default=8192)
    p.add_argument("--seed", type=int, default=20260519)
    args = p.parse_args()

    cache_dir = repo_path(args.cache_dir); out_dir = repo_path(args.out_dir)
    assert cache_dir and out_dir
    paths = sorted(cache_dir.glob("*.npz"))[: args.limit]
    if not paths:
        raise FileNotFoundError(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples: list[dict[str, Any]] = []
    for i, path in enumerate(paths):
        z = np.load(path, allow_pickle=False)
        noisy = z["noisy"].astype(np.float32)
        base = z["base"].astype(np.float32)
        clean = z["clean"].astype(np.float32)
        gs = geometry_stats(noisy, k=args.k, max_points=args.max_stat_points, seed=args.seed + i)
        samples.append({"path": str(path), "noisy": noisy, "base": base, "clean": clean, **gs})
    pr = rank01([s["plane_res_p75"] for s in samples])
    er = rank01([s["edge_conf_p75"] for s in samples])
    for s, a, b in zip(samples, pr, er):
        s["noise_conf"] = float(a)
        s["edge_protect"] = float(b)
        s["gate"] = float(np.clip(s["noise_conf"] * (1.0 - s["edge_protect"]), 0.0, 0.45))

    rules = {
        "identity": {"alpha": 0.0, "max_step_ratio": 0.0, "sign": 0.0},
        "tiny_extend_a002": {"alpha": 0.002, "max_step_ratio": 0.010, "sign": 1.0},
        "tiny_extend_a005": {"alpha": 0.005, "max_step_ratio": 0.015, "sign": 1.0},
        "tiny_shrink_a002": {"alpha": 0.002, "max_step_ratio": 0.010, "sign": -1.0},
        "tiny_shrink_a005": {"alpha": 0.005, "max_step_ratio": 0.015, "sign": -1.0},
    }
    summary = {"cache_dir": str(cache_dir), "n": len(samples), "rules": {}}

    for name, cfg in rules.items():
        rows = []
        for i, s in enumerate(samples):
            if name == "identity":
                pred = s["base"]
                dstats = {"delta_l2_mean": 0.0, "delta_l2_p95": 0.0, "delta_l2_max": 0.0, "clipped_frac": 0.0}
            else:
                pred, dstats = apply_delta(s["noisy"], s["base"], alpha=cfg["alpha"], max_step_ratio=cfg["max_step_ratio"], sign=cfg["sign"], gate=s["gate"], scale_mean=s["scale_mean"])
            cd_noisy = chamfer_l2_np(s["noisy"], s["clean"])
            cd_base = chamfer_l2_np(s["base"], s["clean"])
            cd_pred = chamfer_l2_np(pred, s["clean"])
            rows.append({
                "idx": i,
                "cache": s["path"],
                "rule": name,
                "cd_noisy": cd_noisy,
                "cd_base": cd_base,
                "cd_pred": cd_pred,
                "base_better_than_noisy": int(cd_base < cd_noisy),
                "pred_better_than_base": int(cd_pred < cd_base),
                "pred_better_than_noisy": int(cd_pred < cd_noisy),
                "score_vs_base": metric_to_score(cd_pred, cd_base),
                "gate": s["gate"],
                **dstats,
            })
        with (out_dir / f"eval_{name}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        cd_noisy = [r["cd_noisy"] for r in rows]
        cd_base = [r["cd_base"] for r in rows]
        cd_pred = [r["cd_pred"] for r in rows]
        summary["rules"][name] = {
            "config": cfg,
            "cd_noisy": float(np.mean(cd_noisy)),
            "cd_base": float(np.mean(cd_base)),
            "cd_pred": float(np.mean(cd_pred)),
            "cd_gain_vs_base": float(np.mean(cd_base) - np.mean(cd_pred)),
            "base_better_than_noisy_rate": float(np.mean([r["base_better_than_noisy"] for r in rows])),
            "pred_better_than_base_rate": float(np.mean([r["pred_better_than_base"] for r in rows])),
            "pred_better_than_noisy_rate": float(np.mean([r["pred_better_than_noisy"] for r in rows])),
            "score_vs_base": float(np.mean([r["score_vs_base"] for r in rows])),
            "gate": finite_stats([r["gate"] for r in rows]),
            "delta_l2_mean": finite_stats([r["delta_l2_mean"] for r in rows]),
            "delta_l2_p95": finite_stats([r["delta_l2_p95"] for r in rows]),
            "delta_l2_max": finite_stats([r["delta_l2_max"] for r in rows]),
            "clipped_frac": finite_stats([r["clipped_frac"] for r in rows]),
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    lines = ["# Bounded tiny adapter proxy GT eval", "", f"Cache: `{cache_dir}`", "", "| rule | cd_base | cd_pred | gain | win_rate | delta_mean | note |", "|---|---:|---:|---:|---:|---:|---|"]
    for name, d in summary["rules"].items():
        note = "identity" if name == "identity" else ("shrink" if d["config"]["sign"] < 0 else "extend")
        lines.append(f"| `{name}` | {d['cd_base']:.8g} | {d['cd_pred']:.8g} | {d['cd_gain_vs_base']:.8g} | {d['pred_better_than_base_rate']:.3f} | {d['delta_l2_mean']['mean']:.8g} | {note} |")
    (out_dir / "risk_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
