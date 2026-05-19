#!/usr/bin/env python3
"""Phase-0 candidate evaluation suite for point-cloud denoising artifacts.

This is a lightweight, hidden-test-safe suite. It does not require GT and does
not claim official CD/P2S. It standardizes the bookkeeping that must happen
before training GARA-D:
- zip validity via scripts/check_submission.py;
- SHA256 / size / file inventory;
- optional diagnostics CSV summary from existing blend probes;
- movement analysis against noisy input and optional base zips;
- comparison reports for VM, raw LIR, fixed075, adaptive gates, force-strong, etc.

Official evaluate.py integration can be added when a GT split is available.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import statistics
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# 默认候选是历史/本机审计清单，用于复盘已知 artifact，不代表当前推荐提交。
# 公开复现或新候选应优先通过 --candidate 显式传入，避免依赖个人桌面路径。
DEFAULT_CANDIDATES: dict[str, dict[str, str]] = {
    "official_vm_streaming": {
        "zip": "/home/sallen/Desktop/result_official_vm_streaming_200_smoke_20260516_1648.zip",
        "role": "base_vm",
        "notes": "VM / official_vm_fixed_stitch streaming anchor; official score 48.04.",
    },
    "raw_lir": {
        "zip": "/home/sallen/Desktop/lir_t2_a075_hidden_full_20260517_0555.zip",
        "role": "base_lir",
        "notes": "Raw LIR T=2 alpha=0.75; official score 39.08.",
    },
    "fixed075": {
        "zip": "/home/sallen/Desktop/result_blend_best075_lir025_20260517.zip",
        "role": "fixed_blend_best",
        "notes": "0.75 VM + 0.25 raw LIR; official score 53.32 current internal best.",
    },
    "fixed075_control": {
        "zip": "analysis/p0_geometry_blend_gate_20260518/result_fixed075_control_20260518.zip",
        "role": "identity_fixed075_control",
        "notes": "Regenerated fixed075 control from P0 script; useful for identity/base-chain drift checks.",
    },
    "noise_gate": {
        "zip": "analysis/adaptive_blend_probe_20260517/result_adaptive_blend_noise_gate_20260517.zip",
        "role": "adaptive_blend",
        "notes": "Noisy-only noise_gate adaptive blend; official score 52.57.",
    },
    "p0_plane_balanced": {
        "zip": "analysis/p0_geometry_blend_gate_20260518/result_p0_plane_balanced_20260518.zip",
        "role": "adaptive_blend",
        "notes": "P0 geometry confidence adaptive blend; official score 53.22.",
    },
}

DIAGNOSTIC_HINTS: dict[str, str] = {
    # 若候选生成脚本产出了 diagnostics.csv，就在 suite 报告中汇总关键数值分布。
    "fixed075": "analysis/adaptive_blend_probe_20260517/diagnostics_fixed075.csv",
    "fixed075_control": "analysis/p0_geometry_blend_gate_20260518/diagnostics_fixed075_control.csv",
    "noise_gate": "analysis/adaptive_blend_probe_20260517/diagnostics_noise_gate.csv",
    "p0_plane_balanced": "analysis/p0_geometry_blend_gate_20260518/diagnostics_p0_plane_balanced.csv",
}

OFFICIAL_SCORES: dict[str, dict[str, float]] = {
    # 这里记录的是 leaderboard 已知事实，不是本脚本重新计算出的官方分数。
    "official_vm_streaming": {"score": 48.04, "cd_score": 34.04, "p2s_score": 62.05},
    "raw_lir": {"score": 39.08, "cd_score": 27.53, "p2s_score": 50.63},
    "fixed075": {"score": 53.32, "cd_score": 40.65, "p2s_score": 65.99},
    "noise_gate": {"score": 52.57, "cd_score": 39.67, "p2s_score": 65.48},
    "p0_plane_balanced": {"score": 53.22, "cd_score": 40.55, "p2s_score": 65.90},
}


def repo_path(path: str | Path | None) -> Path | None:
    if path is None or str(path) == "":
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def rel(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def npy_from_zip(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    with zf.open(name, "r") as f:
        return np.load(io.BytesIO(f.read()), allow_pickle=False)


def noisy_path_from_entry(test_root: Path, entry: str) -> Path:
    return test_root / Path(entry).parent / "noisy.npy"


def finite_stats(values: list[float]) -> dict[str, float]:
    xs = [float(x) for x in values if np.isfinite(x)]
    if not xs:
        return {"mean": float("nan"), "p50": float("nan"), "p95": float("nan"), "max": float("nan")}
    arr = np.asarray(xs, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(arr.max()),
    }


def run_check_submission(zip_path: Path, test_root: Path | None, expected_count: int, expected_shape: str) -> dict[str, Any]:
    # 候选 suite 复用正式提交检查脚本，避免两套 zip 验证逻辑不一致。
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "check_submission.py"),
        str(zip_path),
        "--expected-count",
        str(expected_count),
        "--expected-shape",
        expected_shape,
    ]
    if test_root is not None:
        cmd.extend(["--test-root", str(test_root)])
    else:
        cmd.append("--no-test-root-match")
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "command": cmd,
    }


def inventory_zip(zip_path: Path) -> dict[str, Any]:
    # 清点 zip 内部文件、shape、dtype 和有限值范围；这一步不需要 GT，隐藏测试安全。
    shapes: dict[str, int] = {}
    dtypes: dict[str, int] = {}
    names: list[str] = []
    mins: list[float] = []
    maxs: list[float] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in sorted(n for n in zf.namelist() if not n.endswith("/")):
            names.append(name)
            if name.endswith(".npy"):
                arr = npy_from_zip(zf, name)
                shapes[str(tuple(arr.shape))] = shapes.get(str(tuple(arr.shape)), 0) + 1
                dtypes[str(arr.dtype)] = dtypes.get(str(arr.dtype), 0) + 1
                if np.isfinite(arr).all():
                    mins.append(float(arr.min()))
                    maxs.append(float(arr.max()))
    return {
        "file_count": len(names),
        "first_entries": names[:5],
        "shapes": shapes,
        "dtypes": dtypes,
        "finite_min": min(mins) if mins else None,
        "finite_max": max(maxs) if maxs else None,
    }


def movement_vs_noisy(zip_path: Path, test_root: Path, sample_limit: int = 0) -> dict[str, Any]:
    # 对比 denoised 与 noisy 的位移分布，用于发现“移动过大/低噪声过修正”风险。
    # 这不是质量分数，只是提交前的风险体检。
    means: list[float] = []
    p95s: list[float] = []
    maxs: list[float] = []
    missing: list[str] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = sorted(n for n in zf.namelist() if n.endswith("denoised.npy"))
        if sample_limit > 0:
            names = names[:sample_limit]
        for name in names:
            noisy_path = noisy_path_from_entry(test_root, name)
            if not noisy_path.exists():
                missing.append(name)
                continue
            pred = npy_from_zip(zf, name).astype(np.float32, copy=False)
            noisy = np.load(noisy_path, allow_pickle=False).astype(np.float32, copy=False)
            if pred.shape != noisy.shape:
                missing.append(f"shape_mismatch:{name}")
                continue
            disp = np.sqrt(((pred - noisy) ** 2).sum(axis=-1))
            means.append(float(disp.mean()))
            p95s.append(float(np.quantile(disp, 0.95)))
            maxs.append(float(disp.max()))
    return {
        "sample_count": len(means),
        "missing_or_bad": missing[:20],
        "disp_mean": finite_stats(means),
        "disp_p95": finite_stats(p95s),
        "disp_max": finite_stats(maxs),
    }


def movement_vs_base(zip_path: Path, base_zip: Path, sample_limit: int = 0) -> dict[str, Any]:
    # 与 fixed075/VM/LIR 等基准 artifact 做逐点位移比较，判断候选到底改动了多少。
    means: list[float] = []
    p95s: list[float] = []
    maxs: list[float] = []
    bad: list[str] = []
    with zipfile.ZipFile(zip_path, "r") as zf, zipfile.ZipFile(base_zip, "r") as zb:
        names = sorted(n for n in zf.namelist() if n.endswith("denoised.npy"))
        base_names = set(n for n in zb.namelist() if n.endswith("denoised.npy"))
        if sample_limit > 0:
            names = names[:sample_limit]
        for name in names:
            if name not in base_names:
                bad.append(f"missing_base:{name}")
                continue
            arr = npy_from_zip(zf, name).astype(np.float32, copy=False)
            base = npy_from_zip(zb, name).astype(np.float32, copy=False)
            if arr.shape != base.shape:
                bad.append(f"shape_mismatch:{name}")
                continue
            disp = np.sqrt(((arr - base) ** 2).sum(axis=-1))
            means.append(float(disp.mean()))
            p95s.append(float(np.quantile(disp, 0.95)))
            maxs.append(float(disp.max()))
    return {
        "sample_count": len(means),
        "bad": bad[:20],
        "disp_mean": finite_stats(means),
        "disp_p95": finite_stats(p95s),
        "disp_max": finite_stats(maxs),
    }


def read_diagnostics(csv_path: Path) -> dict[str, Any]:
    if not csv_path.exists():
        return {"exists": False}
    # diagnostics 字段很多，只抽取当前审计最关心的噪声/边缘/位移/权重统计。
    numeric: dict[str, list[float]] = {}
    n = 0
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n += 1
            for k, v in row.items():
                if v is None or v == "":
                    continue
                try:
                    x = float(v)
                except ValueError:
                    continue
                numeric.setdefault(k, []).append(x)
    fields = {}
    for k, xs in numeric.items():
        if k in {
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
        }:
            fields[k] = finite_stats(xs)
    return {"exists": True, "path": rel(csv_path), "rows": n, "fields": fields}


def evaluate_candidate(
    name: str,
    info: dict[str, str],
    out_dir: Path,
    test_root: Path | None,
    expected_count: int,
    expected_shape: str,
    base_zips: dict[str, Path],
    sample_limit: int,
) -> dict[str, Any]:
    # 单个候选的评估结果写到 analysis 下，方便之后把有价值结论迁移到 docs/experiments。
    zip_path = repo_path(info["zip"])
    assert zip_path is not None
    payload: dict[str, Any] = {
        "name": name,
        "role": info.get("role", ""),
        "notes": info.get("notes", ""),
        "zip_path": rel(zip_path),
        "exists": zip_path.exists(),
        "official_scores_known": OFFICIAL_SCORES.get(name, {}),
    }
    if not zip_path.exists():
        payload["error"] = "zip not found"
        return payload
    payload["sha256"] = sha256_file(zip_path)
    payload["size_mb"] = round(zip_path.stat().st_size / (1024 * 1024), 3)
    # 先做 artifact 层检查，再做可选位移诊断；任一阶段失败都保留 payload 方便定位。
    payload["inventory"] = inventory_zip(zip_path)
    payload["check_submission"] = run_check_submission(zip_path, test_root, expected_count, expected_shape)
    if test_root is not None and test_root.exists():
        payload["movement_vs_noisy"] = movement_vs_noisy(zip_path, test_root, sample_limit=sample_limit)
    diag_hint = DIAGNOSTIC_HINTS.get(name, "")
    if diag_hint:
        payload["diagnostics"] = read_diagnostics(repo_path(diag_hint) or Path(diag_hint))
    comparisons = {}
    for base_name, base_path in base_zips.items():
        if base_path.exists() and base_path != zip_path:
            comparisons[base_name] = movement_vs_base(zip_path, base_path, sample_limit=sample_limit)
    if comparisons:
        payload["movement_vs_base"] = comparisons
    cand_dir = out_dir / name
    cand_dir.mkdir(parents=True, exist_ok=True)
    (cand_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    (cand_dir / "check_submission.stdout.txt").write_text(payload["check_submission"].get("stdout", ""))
    (cand_dir / "check_submission.stderr.txt").write_text(payload["check_submission"].get("stderr", ""))
    return payload


def write_suite_reports(out_dir: Path, results: list[dict[str, Any]]) -> None:
    # summary.json 给脚本消费，risk_report.md 给人快速决策；两者来自同一 results。
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "suite": "phase0_candidate_suite_v0",
        "count": len(results),
        "results": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Phase 0 Candidate Evaluation Suite",
        "",
        f"Generated: {summary['created_at']}",
        "",
        "## Candidate table",
        "",
        "| name | role | check | official | CD | P2S | sha256 | notes |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for r in results:
        scores = r.get("official_scores_known", {}) or {}
        check = "OK" if r.get("check_submission", {}).get("ok") else "FAIL"
        lines.append(
            "| {name} | {role} | {check} | {score} | {cd} | {p2s} | `{sha}` | {notes} |".format(
                name=r.get("name", ""),
                role=r.get("role", ""),
                check=check,
                score=scores.get("score", ""),
                cd=scores.get("cd_score", ""),
                p2s=scores.get("p2s_score", ""),
                sha=str(r.get("sha256", ""))[:12],
                notes=str(r.get("notes", "")).replace("|", "/"),
            )
        )
    lines.extend([
        "",
        "## Initial risk notes",
        "",
        "- This v0 suite is hidden-test-safe: it validates packages and analyzes movement/diagnostics, but does not claim new official CD/P2S.",
        "- Use official score columns only as recorded leaderboard facts, not as locally recomputed metrics.",
        "- Next extension: add GT split support and bucketed official evaluate.py outputs when local GT/mesh roots are available.",
    ])
    (out_dir / "risk_report.md").write_text("\n".join(lines) + "\n")


def parse_candidate_arg(values: list[str]) -> dict[str, dict[str, str]]:
    if not values:
        return dict(DEFAULT_CANDIDATES)
    # 显式 candidate 参数只需要 name=zip_path，角色默认 custom，避免 CLI 过长。
    out: dict[str, dict[str, str]] = {}
    for item in values:
        parts = item.split("=", 1)
        if len(parts) != 2:
            raise SystemExit(f"--candidate must look like name=/path/to.zip, got: {item}")
        out[parts[0]] = {"zip": parts[1], "role": "custom", "notes": "custom candidate"}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Phase-0 candidate evaluation suite for denoising submission zips.")
    p.add_argument("--candidate", action="append", default=[], help="Candidate override as name=zip_path. If omitted, uses known Phase-0 defaults.")
    p.add_argument("--out-dir", default="analysis/eval_suite_20260518")
    p.add_argument("--test-root", default="dataset_test_noisy")
    p.add_argument("--expected-count", type=int, default=200)
    p.add_argument("--expected-shape", default="50000,3")
    p.add_argument("--sample-limit", type=int, default=0, help="Limit per-point movement analysis to first N samples; 0=all.")
    args = p.parse_args()

    candidates = parse_candidate_arg(args.candidate)
    out_dir = repo_path(args.out_dir) or Path(args.out_dir)
    test_root = repo_path(args.test_root)
    if test_root is not None and not test_root.exists():
        # 没有 noisy root 时仍可做 zip inventory/check_submission 的结构检查。
        print(f"warning: test_root not found, disabling noisy movement/test-root matching: {test_root}", file=sys.stderr)
        test_root = None

    base_zips: dict[str, Path] = {}
    # 只有候选清单里存在这些基准时才做逐点基准比较，避免硬依赖本机历史 artifact。
    for key in ["official_vm_streaming", "raw_lir", "fixed075"]:
        if key in candidates:
            zp = repo_path(candidates[key]["zip"])
            if zp is not None:
                base_zips[key] = zp

    results = []
    for name, info in candidates.items():
        print(f"[suite] evaluating {name}: {info['zip']}")
        results.append(
            evaluate_candidate(
                name=name,
                info=info,
                out_dir=out_dir,
                test_root=test_root,
                expected_count=args.expected_count,
                expected_shape=args.expected_shape,
                base_zips=base_zips,
                sample_limit=args.sample_limit,
            )
        )
    write_suite_reports(out_dir, results)
    print(f"suite summary: {rel(out_dir / 'summary.json')}")
    print(f"risk report:    {rel(out_dir / 'risk_report.md')}")


if __name__ == "__main__":
    main()
