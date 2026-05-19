#!/usr/bin/env python3
"""Automatic Candidate Registry for the formal denoising competition.

This script records candidate submission artifacts in three synchronized files:
- experiments/candidates.jsonl          machine-readable append-only log
- experiments/candidate_registry.csv   spreadsheet-friendly table
- experiments/candidate_registry.md    compact human summary

It intentionally does not train, predict, or evaluate models. It only hashes and
records artifacts that already exist, so it is safe to run after a candidate zip
has been generated and checked.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSONL = ROOT / "experiments" / "candidates.jsonl"
DEFAULT_CSV = ROOT / "experiments" / "candidate_registry.csv"
DEFAULT_MD = ROOT / "experiments" / "candidate_registry.md"
SCHEMA_VERSION = "candidate-registry-v1"

# registry 字段是跨 JSONL/CSV/Markdown 三份输出共享的稳定 schema。
# 追加字段时应只追加不重排，避免旧记录、表格和外部脚本出现列语义漂移。
FIELDS = [
    "schema_version",
    "name",
    "stage",
    "status",
    "created_at",
    "git_commit",
    "git_dirty",
    "config_path",
    "config_sha256",
    "profiles",
    "profile_sha256",
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_size_mb",
    "zip_path",
    "zip_sha256",
    "zip_size_mb",
    "branch",
    "router_threshold",
    "soft_gate_temperature",
    "patch_size",
    "chunk_size",
    "overlap",
    "stitching_strategy",
    "script_version",
    "evaluate_py_path",
    "evaluate_py_sha256",
    "official_eval_json",
    "official_score",
    "cd_score",
    "p2s_score",
    "low_cd",
    "mid_cd",
    "high_cd",
    "latency_sec",
    "peak_memory_mb",
    "large_cloud_stress",
    "params_count",
    "model_size_mb",
    "inference_fps_per_10k_points",
    "inference_ms_per_10k_points",
    "baseline_speedup_ratio",
    "baseline_size_ratio",
    "submission_check",
    "official_submitted",
    "official_score_recorded",
    "conclusion",
    "notes",
]


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


def sha256_file(path: Path | None) -> str:
    if path is None or not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    # 按块读取，避免大 checkpoint / zip 一次性进入内存。
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def size_mb(path: Path | None) -> str:
    if path is None or not path.exists() or not path.is_file():
        return ""
    return f"{path.stat().st_size / (1024 * 1024):.3f}"


def read_yaml(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def deep_update(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_merged_config(config: Path | None, profiles: list[Path]) -> dict[str, Any]:
    cfg = read_yaml(config)
    for profile in profiles:
        cfg = deep_update(cfg, read_yaml(profile))
    return cfg


def git_text(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def git_dirty() -> str:
    # dirty 状态不是失败条件，只是提醒候选 artifact 不是干净 commit 的直接产物。
    status = git_text("status", "--short")
    return "true" if status else "false"


def load_eval_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def load_existing(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    # JSONL 是 append/rebuild 的主数据源；坏行跳过，避免单条手工损坏影响全表重建。
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def normalize_row(row: dict[str, Any]) -> dict[str, str]:
    out = {}
    for field in FIELDS:
        value = row.get(field, "")
        # CSV/Markdown 只能安全承载字符串；复杂对象统一序列化，保持 JSONL 为真源。
        if isinstance(value, (dict, list)):
            out[field] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        elif value is None:
            out[field] = ""
        else:
            out[field] = str(value)
    return out


def write_outputs(rows: list[dict[str, Any]], jsonl: Path, csv_path: Path, md_path: Path) -> None:
    # 三份输出每次一起重写，确保机器可读 JSONL、表格 CSV 和人读 Markdown 不分叉。
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    with jsonl.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(normalize_row(row))

    lines = [
        "# Candidate Registry",
        "",
        "Generated automatically by `scripts/candidate_registry.py`. Do not hand-edit the tables; append via the script.",
        "",
        f"Total candidates: {len(rows)}",
        "",
        "| time | name | stage | status | branch | patch | threshold | CD | P2S | official | zip | conclusion |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        n = normalize_row(row)
        lines.append(
            "| {created_at} | {name} | {stage} | {status} | {branch} | {patch_size} | {router_threshold} | "
            "{cd_score} | {p2s_score} | {official_score} | `{zip_path}` | {conclusion} |".format(**n)
        )
    md_path.write_text("\n".join(lines) + "\n")


def build_row(args: argparse.Namespace) -> dict[str, Any]:
    # 这里只汇总现有 artifact 和配置，不训练、不推理、不改任何 zip。
    # 分数字段优先取 CLI；若传入 official_eval_json，则从 JSON sidecar 补齐。
    config = repo_path(args.config)
    profiles = [repo_path(x) for x in args.profile]
    profiles = [p for p in profiles if p is not None]
    cfg = load_merged_config(config, profiles)
    paths = cfg.get("paths", {})
    pred = cfg.get("predict", {})
    exp = cfg.get("experiment", {})

    ckpt = repo_path(args.ckpt or paths.get("ckpt", ""))
    zip_path = repo_path(args.zip or paths.get("zip", ""))
    evaluate_py = repo_path(args.evaluate_py)
    official_eval = repo_path(args.official_eval_json)
    eval_data = load_eval_json(official_eval)

    name = args.name or exp.get("name") or (zip_path.stem if zip_path else "candidate")
    patch_size = args.patch_size or pred.get("patch_size", "")

    # row 尽量记录“别人复现这个 zip 需要知道什么”，包括代码 commit、配置 hash、
    # checkpoint hash、提交包 hash、路由/patch 参数、已知官方或本地评估结果。
    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "stage": args.stage,
        "status": args.status,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "git_commit": git_text("rev-parse", "HEAD"),
        "git_dirty": git_dirty(),
        "config_path": rel(config),
        "config_sha256": sha256_file(config),
        "profiles": ";".join(rel(p) for p in profiles),
        "profile_sha256": ";".join(sha256_file(p) for p in profiles),
        "checkpoint_path": rel(ckpt),
        "checkpoint_sha256": sha256_file(ckpt),
        "checkpoint_size_mb": size_mb(ckpt),
        "zip_path": rel(zip_path),
        "zip_sha256": sha256_file(zip_path),
        "zip_size_mb": size_mb(zip_path),
        "branch": args.branch,
        "router_threshold": args.router_threshold,
        "soft_gate_temperature": args.soft_gate_temperature,
        "patch_size": patch_size,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "stitching_strategy": args.stitching_strategy,
        "script_version": f"{SCHEMA_VERSION}; python={platform.python_version()}",
        "evaluate_py_path": rel(evaluate_py),
        "evaluate_py_sha256": sha256_file(evaluate_py),
        "official_eval_json": rel(official_eval),
        "official_score": args.official_score or eval_data.get("final_score", ""),
        "cd_score": args.cd_score or eval_data.get("cd_score", ""),
        "p2s_score": args.p2s_score or eval_data.get("p2s_score", ""),
        "low_cd": args.low_cd,
        "mid_cd": args.mid_cd,
        "high_cd": args.high_cd,
        "latency_sec": args.latency_sec,
        "peak_memory_mb": args.peak_memory_mb,
        "large_cloud_stress": args.large_cloud_stress,
        "params_count": args.params_count,
        "model_size_mb": args.model_size_mb or size_mb(ckpt),
        "inference_fps_per_10k_points": args.inference_fps_per_10k_points,
        "inference_ms_per_10k_points": args.inference_ms_per_10k_points,
        "baseline_speedup_ratio": args.baseline_speedup_ratio,
        "baseline_size_ratio": args.baseline_size_ratio,
        "submission_check": args.submission_check,
        "official_submitted": args.official_submitted,
        "official_score_recorded": "true" if (args.official_score or eval_data.get("final_score")) else "false",
        "conclusion": args.conclusion,
        "notes": args.notes,
    }
    return row


def main() -> None:
    p = argparse.ArgumentParser(description="Append or rebuild the formal Candidate Registry.")
    # registry 脚本的默认模式是追加候选；--rebuild-only 只从 JSONL 重建 CSV/MD。
    p.add_argument("--name", default="")
    p.add_argument("--stage", default="Phase 1")
    p.add_argument("--status", default="candidate", choices=["candidate", "submitted", "rejected", "baseline", "archive"])
    p.add_argument("--config", default="configs/denoise_baseline.yaml")
    p.add_argument("--profile", action="append", default=[])
    p.add_argument("--ckpt", default="")
    p.add_argument("--zip", default="")
    p.add_argument("--branch", default="")
    p.add_argument("--router-threshold", default="")
    p.add_argument("--soft-gate-temperature", default="")
    p.add_argument("--patch-size", default="")
    p.add_argument("--chunk-size", default="")
    p.add_argument("--overlap", default="0")
    p.add_argument("--stitching-strategy", default="contiguous_chunks")
    p.add_argument("--evaluate-py", default="starter_code/evaluate.py")
    p.add_argument("--official-eval-json", default="")
    p.add_argument("--official-score", default="")
    p.add_argument("--cd-score", default="")
    p.add_argument("--p2s-score", default="")
    p.add_argument("--low-cd", default="")
    p.add_argument("--mid-cd", default="")
    p.add_argument("--high-cd", default="")
    p.add_argument("--latency-sec", default="")
    p.add_argument("--peak-memory-mb", default="")
    p.add_argument("--large-cloud-stress", default="")
    p.add_argument("--params-count", default="")
    p.add_argument("--model-size-mb", default="")
    p.add_argument("--inference-fps-per-10k-points", default="")
    p.add_argument("--inference-ms-per-10k-points", default="")
    p.add_argument("--baseline-speedup-ratio", default="")
    p.add_argument("--baseline-size-ratio", default="")
    p.add_argument("--submission-check", default="not_run")
    p.add_argument("--official-submitted", default="false")
    p.add_argument("--conclusion", default="")
    p.add_argument("--notes", default="")
    p.add_argument("--jsonl", default=str(DEFAULT_JSONL))
    p.add_argument("--csv", default=str(DEFAULT_CSV))
    p.add_argument("--md", default=str(DEFAULT_MD))
    p.add_argument("--rebuild-only", action="store_true", help="Rebuild CSV/MD from existing JSONL without appending")
    args = p.parse_args()

    jsonl = repo_path(args.jsonl) or DEFAULT_JSONL
    csv_path = repo_path(args.csv) or DEFAULT_CSV
    md_path = repo_path(args.md) or DEFAULT_MD
    rows = load_existing(jsonl)
    if not args.rebuild_only:
        # 不做去重：同名候选多次登记时保留完整时间线，后续按 created_at/sha 区分。
        rows.append(build_row(args))
    write_outputs(rows, jsonl, csv_path, md_path)
    print(f"registry rows: {len(rows)}")
    print(f"jsonl: {rel(jsonl)}")
    print(f"csv:   {rel(csv_path)}")
    print(f"md:    {rel(md_path)}")


if __name__ == "__main__":
    main()
