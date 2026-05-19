#!/usr/bin/env python3
"""GARA-D bounded tiny artifact adapter smoke.

Hidden-test-safe: no GT, no training. Applies tiny, bounded residual deltas to a
fixed075 base artifact using only noisy/base geometry. This is a risk smoke, not
a submission recommendation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
# No machine-local default: pass --base-zip explicitly to avoid hidden personal paths in reproducible runs.
DEFAULT_BASE = Path("")

# 每个输出点云的几何统计、门控权重和实际 delta 分布都会写入 diagnostics CSV。
FIELDS = [
    "entry", "category", "model_id", "n_points", "rule",
    "plane_res_p75", "edge_conf_mean", "edge_conf_p75", "scale_mean",
    "noise_conf", "edge_protect", "gate", "alpha", "max_step_ratio",
    "base_move_mean", "base_move_p95", "delta_l2_mean", "delta_l2_p95", "delta_l2_max",
    "delta_over_base_move_mean", "clipped_frac",
]


@dataclass(frozen=True)
class Rule:
    # 这些 rule 是 smoke 风险探针，不是提交策略；alpha/max_step_ratio 都故意很小。
    name: str
    alpha: float
    max_step_ratio: float
    min_gate: float = 0.0
    max_gate: float = 0.45
    direction_sign: float = 1.0  # + extends base-noisy vector; - shrinks toward noisy


def repo_path(path: str | Path | None) -> Path | None:
    if path is None or str(path) == "":
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


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
    return test_root / Path(entry).parent / "noisy.npy"


def sample_for_stats(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) <= max_points:
        return pts
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts), max_points, replace=False)
    return pts[idx]


def geometry_stats(points: np.ndarray, *, k: int, max_points: int, seed: int) -> dict[str, float]:
    # 只从 noisy 点云估计粗糙度/边缘性，保证隐藏测试安全，不读取 clean/GT。
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
        "edge_conf_mean": float(edge_conf.mean()),
        "edge_conf_p75": float(np.quantile(edge_conf, 0.75)),
        "scale_mean": float(scale.mean()),
    }


def rank01(values: list[float]) -> np.ndarray:
    # 使用全批次 rank 而不是绝对阈值，降低不同数据尺度下规则失效的风险。
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    out = np.empty_like(arr)
    if len(arr) == 1:
        out[0] = 0.5
    else:
        out[order] = np.linspace(0.0, 1.0, len(arr))
    return out


def finite_stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "p50": float(np.quantile(arr, .5)), "p95": float(np.quantile(arr, .95)), "max": float(arr.max())}


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ["gate", "base_move_mean", "base_move_p95", "delta_l2_mean", "delta_l2_p95", "delta_l2_max", "delta_over_base_move_mean", "clipped_frac"]
    return {k: finite_stats([float(r[k]) for r in rows]) for k in keys}


def write_report(out_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# GARA-D bounded tiny artifact adapter smoke",
        "",
        "Hidden-test-safe: no GT and no training. This report only checks whether tiny bounded deltas can pass packaging/risk gates.",
        "",
        "| rule | alpha | sign | max_step_ratio | gate mean/p95 | delta mean | delta p95 | delta max | clipped mean | note |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, d in summary["rules"].items():
        risk = d["risk"]
        cfg = d["config"]
        note = "extends base-noisy vector" if cfg["direction_sign"] > 0 else "shrinks toward noisy"
        lines.append(
            f"| `{name}` | {cfg['alpha']:.5f} | {cfg['direction_sign']:+.0f} | {cfg['max_step_ratio']:.5f} | "
            f"{risk['gate']['mean']:.4f}/{risk['gate']['p95']:.4f} | "
            f"{risk['delta_l2_mean']['mean']:.8f} | {risk['delta_l2_p95']['mean']:.8f} | {risk['delta_l2_max']['max']:.8f} | "
            f"{risk['clipped_frac']['mean']:.4f} | {note} |"
        )
    lines += [
        "",
        "## Decision rule",
        "",
        "- Passing package/movement risk is necessary but not sufficient for any official submission.",
        "- If movement tails are too large or low-risk samples are disturbed, fix the adapter boundary before training.",
        "- These artifacts are smoke artifacts, not candidate recommendations.",
    ]
    (out_dir / "risk_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-zip", required=True, help="fixed075/base submission zip; pass explicitly, no machine-local default")
    p.add_argument("--test-root", default="dataset_test_noisy")
    p.add_argument("--out-dir", default="analysis/garad_bounded_tiny_fixed075_20260519")
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--max-stat-points", type=int, default=8192)
    p.add_argument("--seed", type=int, default=20260519)
    args = p.parse_args()

    base_zip = repo_path(args.base_zip)
    test_root = repo_path(args.test_root)
    out_dir = repo_path(args.out_dir)
    assert base_zip is not None and test_root is not None and out_dir is not None
    if not base_zip.exists():
        raise FileNotFoundError(base_zip)
    if not test_root.exists():
        raise FileNotFoundError(test_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    rules = [
        Rule("tiny_extend_a002", alpha=0.002, max_step_ratio=0.010, direction_sign=1.0),
        Rule("tiny_extend_a005", alpha=0.005, max_step_ratio=0.015, direction_sign=1.0),
        Rule("tiny_shrink_a002", alpha=0.002, max_step_ratio=0.010, direction_sign=-1.0),
    ]

    base_rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(base_zip, "r") as zb:
        names = sorted(n for n in zb.namelist() if n.endswith("denoised.npy"))
        for i, name in enumerate(names):
            # 先对所有样本收集 noisy/base 统计，再统一做 rank 归一化。
            noisy = np.load(noisy_path_from_entry(test_root, name), allow_pickle=False).astype(np.float32, copy=False)
            base = npy_from_zip(zb, name).astype(np.float32, copy=False)
            if noisy.shape != base.shape:
                raise RuntimeError(f"shape mismatch {name}: {noisy.shape} vs {base.shape}")
            gs = geometry_stats(noisy, k=args.k, max_points=args.max_stat_points, seed=args.seed + i)
            move = np.sqrt(((base - noisy) ** 2).sum(axis=-1))
            parts = Path(name).parts
            base_rows.append({
                "entry": name,
                "category": parts[1] if len(parts) >= 4 else "",
                "model_id": parts[2] if len(parts) >= 4 else "",
                "n_points": int(base.shape[0]),
                **gs,
                "base_move_mean": float(move.mean()),
                "base_move_p95": float(np.quantile(move, .95)),
            })
    plane_rank = rank01([r["plane_res_p75"] for r in base_rows])
    edge_rank = rank01([r["edge_conf_p75"] for r in base_rows])
    for r, pr, er in zip(base_rows, plane_rank, edge_rank):
        # Noisy-only confidence: move more on high roughness/noise, less on edge-like shapes.
        noise_conf = float(pr)
        edge_protect = float(er)
        r["noise_conf"] = noise_conf
        r["edge_protect"] = edge_protect

    summary: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "base_zip": str(base_zip),
        "base_sha256": sha256(base_zip),
        "test_root": str(test_root),
        "method": "GARA-D bounded tiny artifact adapter smoke; no GT/no training",
        "rules": {},
    }

    for rule in rules:
        # 每条 rule 单独生成 zip + diagnostics，方便候选 suite 独立检查和比较。
        zip_path = out_dir / f"result_garad_bounded_{rule.name}_20260519.zip"
        csv_path = out_dir / f"diagnostics_{rule.name}.csv"
        rows: list[dict[str, Any]] = []
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(base_zip, "r") as zb, zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zout, csv_path.open("w", newline="") as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=FIELDS)
            writer.writeheader()
            for br in base_rows:
                name = br["entry"]
                noisy = np.load(noisy_path_from_entry(test_root, name), allow_pickle=False).astype(np.float32, copy=False)
                base = npy_from_zip(zb, name).astype(np.float32, copy=False)
                direction = base - noisy
                direction_l2 = np.sqrt(np.maximum((direction * direction).sum(axis=-1, keepdims=True), 1e-18))
                unit = direction / direction_l2
                # delta 方向沿 base-noisy 向量；extend/shrink 只通过 direction_sign 区分。
                # gate 越高，代表 noisy 更像高噪声且边缘保护较弱。
                gate = np.clip(br["noise_conf"] * (1.0 - br["edge_protect"]), rule.min_gate, rule.max_gate)
                raw_step = rule.alpha * gate * direction_l2[..., 0]
                max_step = rule.max_step_ratio * max(float(br["scale_mean"]), 1e-8)
                # 每点移动还要受局部尺度上限约束，防止少数点产生过长尾位移。
                step = np.minimum(raw_step, max_step)
                delta = (rule.direction_sign * step[:, None] * unit).astype(np.float32)
                out = (base + delta).astype(np.float32)
                write_npy_to_zip(zout, name, out)
                dl2 = np.sqrt(np.maximum((delta * delta).sum(axis=-1), 0.0))
                row = dict(br)
                row.update({
                    "rule": rule.name,
                    "gate": float(gate),
                    "alpha": rule.alpha,
                    "max_step_ratio": rule.max_step_ratio,
                    "delta_l2_mean": float(dl2.mean()),
                    "delta_l2_p95": float(np.quantile(dl2, .95)),
                    "delta_l2_max": float(dl2.max()),
                    "delta_over_base_move_mean": float(dl2.mean() / max(float(br["base_move_mean"]), 1e-12)),
                    "clipped_frac": float((raw_step > max_step).mean()),
                })
                writer.writerow({k: row[k] for k in FIELDS})
                rows.append(row)
        summary["rules"][rule.name] = {
            "zip": str(zip_path),
            "sha256": sha256(zip_path),
            "diagnostics_csv": str(csv_path),
            "config": rule.__dict__,
            "risk": summarize_rows(rows),
        }

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    write_report(out_dir, summary)
    print(json.dumps({"out_dir": rel(out_dir), "rules": {k: {"zip": rel(Path(v["zip"])), "sha256": v["sha256"], "delta_mean": v["risk"]["delta_l2_mean"]["mean"], "delta_p95": v["risk"]["delta_l2_p95"]["mean"], "delta_max": v["risk"]["delta_l2_max"]["max"]} for k, v in summary["rules"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
