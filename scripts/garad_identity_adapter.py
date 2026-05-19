#!/usr/bin/env python3
"""GARA-D fixed075-base identity adapter prototype.

This is deliberately an identity-only generator for Phase 1.5 chain validation.
It establishes the exact packaging path that a future trained GARA-D adapter will
use, while applying delta=0 to the fixed075 base prediction.

Expected validation:
    generated zip vs fixed075 zip should have per-point displacement exactly 0
    (or only explainable np.save/zip metadata differences in file hash).
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
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# No machine-local default: pass --base-zip explicitly to avoid hidden personal paths in reproducible runs.
DEFAULT_BASE = Path("")

# diagnostics 字段只度量 identity adapter 是否真的“零改动”，不是质量评分。
FIELDS = [
    "entry",
    "category",
    "model_id",
    "n_points",
    "base_abs_mean",
    "delta_l2_mean",
    "delta_l2_p95",
    "delta_l2_max",
]


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
    # 用内存 buffer 写 npy，避免先落盘再二次打包产生中间文件。
    buf = io.BytesIO()
    np.save(buf, arr.astype(np.float32, copy=False), allow_pickle=False)
    zf.writestr(name, buf.getvalue())


def zero_delta_adapter(base: np.ndarray) -> np.ndarray:
    """Future GARA-D adapter boundary: return residual delta for base prediction."""
    # identity 版本故意返回 0，用来验证 base zip -> adapter -> output zip 的链路不漂移。
    return np.zeros_like(base, dtype=np.float32)


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(arr.max()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Generate fixed075-base identity adapter submission zip.")
    p.add_argument("--base-zip", required=True, help="fixed075/base submission zip; pass explicitly, no machine-local default")
    p.add_argument("--out-dir", default="analysis/garad_identity_fixed075_20260518")
    p.add_argument("--zip-name", default="result_garad_identity_fixed075_20260518.zip")
    args = p.parse_args()

    base_zip = repo_path(args.base_zip)
    out_dir = repo_path(args.out_dir)
    assert base_zip is not None and out_dir is not None
    if not base_zip.exists():
        raise FileNotFoundError(base_zip)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_zip = out_dir / args.zip_name
    diag_csv = out_dir / "diagnostics_identity.csv"
    rows: list[dict[str, Any]] = []
    delta_means: list[float] = []
    delta_p95s: list[float] = []
    delta_maxs: list[float] = []

    if out_zip.exists():
        out_zip.unlink()

    # 逐条复制 base prediction，并通过 adapter 边界显式加 delta；
    # 如果最后位移不是 0，问题一定在读写/打包链路而不是模型。
    with zipfile.ZipFile(base_zip, "r") as zin, zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zout, diag_csv.open("w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=FIELDS)
        writer.writeheader()
        names = sorted(n for n in zin.namelist() if n.endswith("denoised.npy"))
        if not names:
            raise RuntimeError(f"no denoised.npy entries in {base_zip}")
        for name in names:
            base = npy_from_zip(zin, name).astype(np.float32, copy=False)
            delta = zero_delta_adapter(base)
            out = (base + delta).astype(np.float32, copy=False)
            write_npy_to_zip(zout, name, out)
            d = np.sqrt(((out - base) ** 2).sum(axis=-1))
            parts = Path(name).parts
            row = {
                "entry": name,
                "category": parts[1] if len(parts) >= 4 else "",
                "model_id": parts[2] if len(parts) >= 4 else "",
                "n_points": int(out.shape[0]),
                "base_abs_mean": float(np.abs(base).mean()),
                "delta_l2_mean": float(d.mean()),
                "delta_l2_p95": float(np.quantile(d, 0.95)),
                "delta_l2_max": float(d.max()),
            }
            writer.writerow(row)
            rows.append(row)
            delta_means.append(row["delta_l2_mean"])
            delta_p95s.append(row["delta_l2_p95"])
            delta_maxs.append(row["delta_l2_max"])

    summary = {
        # sha256 同时记录 base/out，方便判断内容漂移还是 zip 元数据差异。
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "method": "GARA-D / 几何定向精炼适配器 identity adapter",
        "adapter_mode": "identity_delta_zero",
        "base_zip": str(base_zip),
        "base_sha256": sha256(base_zip),
        "out_zip": str(out_zip),
        "out_sha256": sha256(out_zip),
        "diagnostics_csv": str(diag_csv),
        "n_entries": len(rows),
        "delta_l2_mean": summarize(delta_means),
        "delta_l2_p95": summarize(delta_p95s),
        "delta_l2_max": summarize(delta_maxs),
        "validation_expectation": "All delta stats should be 0; zip sha may differ from base because np.save/zip metadata can be regenerated.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    report = [
        "# GARA-D fixed075-base identity adapter validation",
        "",
        "This artifact applies `delta = 0` to the fixed075 base prediction through the future GARA-D adapter boundary.",
        "",
        f"- base zip: `{rel(base_zip)}`",
        f"- output zip: `{rel(out_zip)}`",
        f"- entries: {len(rows)}",
        f"- delta_l2_mean.mean: {summary['delta_l2_mean']['mean']:.12g}",
        f"- delta_l2_p95.max: {summary['delta_l2_p95']['max']:.12g}",
        f"- delta_l2_max.max: {summary['delta_l2_max']['max']:.12g}",
        "",
        "Decision rule: if candidate-suite comparison vs fixed075 is non-zero, fix packaging/base chain before any training.",
    ]
    (out_dir / "risk_report.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"out_zip": rel(out_zip), "summary": rel(out_dir / "summary.json"), "delta_l2_max": summary["delta_l2_max"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
