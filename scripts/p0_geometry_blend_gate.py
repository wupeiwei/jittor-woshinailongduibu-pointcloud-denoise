#!/usr/bin/env python3
"""P0 v1 geometry-confidence adaptive blend gate.

No training, no submission. Builds hidden-test submission zips by blending the
known stream/VM output and LIR output with noisy-only geometry confidence.

Main design after noise_gate official result (52.57 < fixed075 53.32):
- keep the mean blend close to fixed075, do not globally reduce LIR exposure;
- redistribute LIR exposure by noisy-only plane residual / roughness;
- keep bounded weights to avoid overfitting the hidden leaderboard;
- emit diagnostics and risk report for candidate review.
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

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
# 这两个 zip 是历史已知基准；脚本只做再加权，不训练、不读 GT。
DEFAULT_STREAM = Path("")
DEFAULT_LIR = Path("")

# 输出字段围绕 noisy-only 几何统计和混合权重展开，便于比较不同 rule 的 exposure 分布。
FIELDS = [
    "entry", "category", "model_id", "rule", "n_points",
    "plane_res_p50", "plane_res_p75", "plane_res_p90", "plane_res_iqr",
    "edge_conf_mean", "edge_conf_p75", "scattering_mean", "scale_mean",
    "plane_rank", "edge_rank", "rough_rank", "stream_weight", "lir_weight",
    "delta_stream_vs_fixed075", "disp_from_stream_mean", "disp_from_stream_p95",
    "disp_from_lir_mean", "disp_from_lir_p95",
]

@dataclass(frozen=True)
class Rule:
    # 这些参数控制权重重分配强度；min/max_w 保证不会偏离 fixed075 太远。
    name: str
    beta_plane: float = 0.0
    beta_edge: float = 0.0
    beta_rough: float = 0.0
    min_w: float = 0.70
    max_w: float = 0.80
    recenter_mean: float = 0.75


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
    # 只用 noisy 端几何特征估计 blend 权重，避免把 GT 信息泄漏进候选规则。
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
        "plane_res_p50": float(np.quantile(plane_res, 0.50)),
        "plane_res_p75": float(np.quantile(plane_res, 0.75)),
        "plane_res_p90": float(np.quantile(plane_res, 0.90)),
        "plane_res_iqr": float(np.quantile(plane_res, 0.75) - np.quantile(plane_res, 0.25)),
        "edge_conf_mean": float(edge_conf.mean()),
        "edge_conf_p75": float(np.quantile(edge_conf, 0.75)),
        "scattering_mean": float(scattering.mean()),
        "scale_mean": float(scale.mean()),
    }


def rank01(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty_like(arr, dtype=np.float64)
    if len(arr) == 1:
        ranks[0] = 0.5
    else:
        ranks[order] = np.linspace(0.0, 1.0, len(arr))
    return ranks.tolist()


def apply_rule(rule: Rule, rows: list[dict[str, float]]) -> list[float]:
    # 先按 noisy-only rank 生成原始权重，再回中到 recenter_mean，尽量贴近 fixed075。
    raw = []
    for r in rows:
        # Higher plane/roughness => more LIR (lower stream weight).
        # Higher edge confidence => more stream protection (higher stream weight).
        score = (
            -rule.beta_plane * (r["plane_rank"] - 0.5)
            + rule.beta_edge * (r["edge_rank"] - 0.5)
            - rule.beta_rough * (r["rough_rank"] - 0.5)
        )
        raw.append(0.75 + score)
    arr = np.asarray(raw, dtype=np.float64)
    # Recenter before clipping so mean exposure stays close to fixed075.
    arr = arr + (rule.recenter_mean - float(arr.mean()))
    arr = np.clip(arr, rule.min_w, rule.max_w)
    # A second gentle recenter if clipping did not make it impossible.
    arr = arr + (rule.recenter_mean - float(arr.mean()))
    arr = np.clip(arr, rule.min_w, rule.max_w)
    return arr.tolist()


def q(values: list[float], p: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), p))


def summarize_rule(rows: list[dict[str, float]]) -> dict:
    # 报告除了全局均值，还按 plane_res 分桶，专门看低/中/高粗糙度样本的曝光是否失衡。
    weights = [r["stream_weight"] for r in rows]
    lir = [r["lir_weight"] for r in rows]
    disp = [r["disp_from_stream_mean"] for r in rows]
    disp95 = [r["disp_from_stream_p95"] for r in rows]
    plane = [r["plane_res_p75"] for r in rows]
    t1, t2 = q(plane, 1/3), q(plane, 2/3)
    def bucket(r):
        if r["plane_res_p75"] <= t1: return "low_res"
        if r["plane_res_p75"] <= t2: return "mid_res"
        return "high_res"
    buckets = {}
    for b in ["low_res", "mid_res", "high_res"]:
        br = [r for r in rows if bucket(r) == b]
        buckets[b] = {
            "n": len(br),
            "stream_weight_mean": float(np.mean([r["stream_weight"] for r in br])),
            "lir_weight_mean": float(np.mean([r["lir_weight"] for r in br])),
            "disp_from_stream_mean": float(np.mean([r["disp_from_stream_mean"] for r in br])),
        }
    pack_keys = ["entry", "plane_res_p75", "edge_conf_p75", "stream_weight", "lir_weight", "disp_from_stream_p95"]
    return {
        "n": len(rows),
        "stream_weight": {"mean": float(np.mean(weights)), "min": min(weights), "p10": q(weights, .1), "p50": q(weights, .5), "p90": q(weights, .9), "max": max(weights)},
        "lir_weight": {"mean": float(np.mean(lir)), "min": min(lir), "p10": q(lir, .1), "p50": q(lir, .5), "p90": q(lir, .9), "max": max(lir)},
        "disp_from_stream_mean": {"mean": float(np.mean(disp)), "p50": q(disp, .5), "p95": q(disp, .95), "max": max(disp)},
        "disp_from_stream_p95": {"mean": float(np.mean(disp95)), "p50": q(disp95, .5), "p95": q(disp95, .95), "max": max(disp95)},
        "buckets": buckets,
        "top_more_lir": [{k: r[k] for k in pack_keys} for r in sorted(rows, key=lambda r: r["stream_weight"])[:10]],
        "top_more_stream": [{k: r[k] for k in pack_keys} for r in sorted(rows, key=lambda r: r["stream_weight"], reverse=True)[:10]],
        "bucket_thresholds": {"low_mid_plane_res_p75": t1, "mid_high_plane_res_p75": t2},
    }


def write_report(out_dir: Path, summary: dict) -> None:
    # Markdown 只做快速人工审查，不替代 summary.json 的机器消费用途。
    lines = [
        "# P0 v1 geometry-confidence adaptive blend gate report",
        "",
        "No training / no official submission. Hidden-test artifact generation only.",
        "Design: keep mean stream weight close to fixed075 while redistributing LIR exposure by noisy-only plane residual / roughness / edge confidence.",
        "",
        "| rule | mean stream_w | min/max stream_w | mean LIR_w | low_res LIR_w | mid_res LIR_w | high_res LIR_w | mean disp_from_stream | max p95 disp | note |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    notes = {
        "p0_plane_balanced": "rank plane-residual redistribution; mean kept at fixed075",
        "p0_plane_edge_safe": "plane residual + edge protection; P2S-safer than pure plane",
        "p0_rough_edge_balanced": "adds roughness/IQR signal; slightly richer noisy-only v1",
    }
    for name, d in summary["rules"].items():
        if name == "fixed075_control":
            note = "control"
        else:
            note = notes.get(name, "candidate")
        b = d["risk"]["buckets"]
        lines.append(
            f"| `{name}` | {d['risk']['stream_weight']['mean']:.6f} | "
            f"{d['risk']['stream_weight']['min']:.6f}/{d['risk']['stream_weight']['max']:.6f} | "
            f"{d['risk']['lir_weight']['mean']:.6f} | "
            f"{b['low_res']['lir_weight_mean']:.6f} | {b['mid_res']['lir_weight_mean']:.6f} | {b['high_res']['lir_weight_mean']:.6f} | "
            f"{d['risk']['disp_from_stream_mean']['mean']:.6f} | {d['risk']['disp_from_stream_p95']['max']:.6f} | {note} |"
        )
    lines += [
        "",
        "## Decision hints",
        "",
        "- `noise_gate` official result was below fixed075, likely because it globally reduced LIR exposure (mean LIR about 0.2018).",
        "- P0 v1 candidates here intentionally keep mean LIR around 0.25 and only redistribute exposure.",
        "- No candidate is a submission recommendation until zip checks, diagnostics review, and risk discussion are complete.",
        "- If one candidate is submitted later, prefer the one with bounded low_res LIR and no aggressive tail.",
    ]
    (out_dir / "risk_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stream-zip", required=True, help="stream/VM submission zip; pass explicitly, no machine-local default")
    p.add_argument("--lir-zip", required=True, help="LIR submission zip; pass explicitly, no machine-local default")
    p.add_argument("--test-root", default="dataset_test_noisy")
    p.add_argument("--out-dir", default="analysis/p0_geometry_blend_gate_20260518")
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--max-stat-points", type=int, default=8192)
    p.add_argument("--seed", type=int, default=20260518)
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

    rules = [
        Rule("fixed075_control", min_w=0.75, max_w=0.75),
        Rule("p0_plane_balanced", beta_plane=0.050, min_w=0.70, max_w=0.80),
        Rule("p0_plane_edge_safe", beta_plane=0.045, beta_edge=0.020, min_w=0.70, max_w=0.81),
        Rule("p0_rough_edge_balanced", beta_plane=0.035, beta_rough=0.020, beta_edge=0.020, min_w=0.70, max_w=0.81),
    ]

    with zipfile.ZipFile(stream_zip, "r") as zs, zipfile.ZipFile(lir_zip, "r") as zl:
        names_s = sorted(n for n in zs.namelist() if n.endswith("denoised.npy"))
        names_l = sorted(n for n in zl.namelist() if n.endswith("denoised.npy"))
        if names_s != names_l:
            raise RuntimeError("stream/lir entries mismatch")

        base_rows: list[dict[str, float]] = []
        for i, name in enumerate(names_s):
            noisy_path = noisy_path_from_entry(test_root, name)
            noisy = np.load(noisy_path, allow_pickle=False)
            s = geometry_stats(noisy, k=args.k, max_points=args.max_stat_points, seed=args.seed + i)
            parts = Path(name).parts
            row = {
                "entry": name,
                "category": parts[1] if len(parts) >= 4 else "",
                "model_id": parts[2] if len(parts) >= 4 else "",
                "n_points": int(noisy.shape[0]),
                **s,
            }
            base_rows.append(row)
        plane_ranks = rank01([r["plane_res_p75"] for r in base_rows])
        edge_ranks = rank01([r["edge_conf_p75"] for r in base_rows])
        rough_ranks = rank01([r["plane_res_iqr"] for r in base_rows])
        for r, pr, er, rr in zip(base_rows, plane_ranks, edge_ranks, rough_ranks):
            r["plane_rank"] = pr
            r["edge_rank"] = er
            r["rough_rank"] = rr

        summary = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "stream_zip": str(stream_zip),
            "stream_sha256": sha256(stream_zip),
            "lir_zip": str(lir_zip),
            "lir_sha256": sha256(lir_zip),
            "test_root": str(test_root),
            "rules": {},
        }

        for rule in rules:
            # 逐 rule 独立输出 zip 和 diagnostics，避免不同候选之间互相污染。
            weights = apply_rule(rule, base_rows)
            rows_out = []
            zip_path = out_dir / f"result_{rule.name}_20260518.zip"
            csv_path = out_dir / f"diagnostics_{rule.name}.csv"
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zout, csv_path.open("w", newline="") as fcsv:
                writer = csv.DictWriter(fcsv, fieldnames=FIELDS)
                writer.writeheader()
                for base, w, name in zip(base_rows, weights, names_s):
                    stream = npy_from_zip(zs, name).astype(np.float32, copy=False)
                    lir = npy_from_zip(zl, name).astype(np.float32, copy=False)
                    out = (w * stream + (1.0 - w) * lir).astype(np.float32)
                    write_npy_to_zip(zout, name, out)
                    ds = np.sqrt(((out - stream) ** 2).sum(axis=-1))
                    dl = np.sqrt(((out - lir) ** 2).sum(axis=-1))
                    row = dict(base)
                    row.update({
                        "rule": rule.name,
                        "stream_weight": float(w),
                        "lir_weight": float(1.0 - w),
                        "delta_stream_vs_fixed075": float(w - 0.75),
                        "disp_from_stream_mean": float(ds.mean()),
                        "disp_from_stream_p95": float(np.quantile(ds, 0.95)),
                        "disp_from_lir_mean": float(dl.mean()),
                        "disp_from_lir_p95": float(np.quantile(dl, 0.95)),
                    })
                    writer.writerow({k: row[k] for k in FIELDS})
                    rows_out.append(row)
            summary["rules"][rule.name] = {
                "zip": str(zip_path),
                "sha256": sha256(zip_path),
                "diagnostics_csv": str(csv_path),
                "risk": summarize_rule(rows_out),
            }

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    write_report(out_dir, summary)
    print(json.dumps({"out_dir": str(out_dir), "rules": {k: {"zip": v["zip"], "sha256": v["sha256"], "mean_stream": v["risk"]["stream_weight"]["mean"], "min_stream": v["risk"]["stream_weight"]["min"], "max_stream": v["risk"]["stream_weight"]["max"]} for k, v in summary["rules"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
