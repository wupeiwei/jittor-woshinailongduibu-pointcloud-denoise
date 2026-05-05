#!/usr/bin/env python3
"""Collect denoising experiment run summaries into a CSV table.

Reads experiments/runs/* produced by scripts/train.sh and extracts:
- config/profile/run metadata;
- final training metrics from train.log;
- resolved config values from config.final.yaml/config.yaml + profile.yaml;
- checkpoint/zip existence when paths are available.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = ROOT / "experiments" / "runs"
DEFAULT_OUT = ROOT / "experiments" / "summary.csv"

METRIC_RE = re.compile(
    r"step=(?P<step>\d+)\s+"
    r"loss=(?P<loss>[-+0-9.eE]+)\s+"
    r"offset_mse=(?P<offset_mse>[-+0-9.eE]+)\s+"
    r"cd=(?P<cd>[-+0-9.eE]+)\s+"
    r"pred_offset_abs_mean=(?P<pred_offset_abs_mean>[-+0-9.eE]+)\s+"
    r"pred_offset_l2_mean=(?P<pred_offset_l2_mean>[-+0-9.eE]+)\s+"
    r"elapsed=(?P<elapsed_sec>[-+0-9.eE]+)s"
)

FIELDS = [
    "run_id",
    "status",
    "date",
    "experiment",
    "config",
    "profile",
    "python",
    "git_dirty",
    "steps_config",
    "final_step",
    "final_loss",
    "final_offset_mse",
    "final_cd",
    "final_pred_offset_abs_mean",
    "final_pred_offset_l2_mean",
    "elapsed_sec",
    "batch_size",
    "num_points",
    "limit",
    "lr",
    "cd_weight",
    "k",
    "feat_dim",
    "hidden",
    "pwsenel",
    "staas",
    "staas_strength",
    "ckpt",
    "ckpt_exists",
    "zip",
    "zip_exists",
    "out_dir",
    "notes",
]


def deep_update(base: dict[str, Any], patch: dict[str, Any] | None) -> dict[str, Any]:
    for key, value in (patch or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def display_path(path: str | Path | None) -> str:
    if path is None or str(path) == "":
        return ""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def repo_path(path: str | Path | None) -> Path | None:
    if path is None or str(path) == "":
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def read_meta(path: Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    if not path.exists():
        return meta
    for line in path.read_text(errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()
    return meta


def git_dirty_from_meta(path: Path) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    dirty = [line for line in lines if line.startswith((" M", "M ", "A ", "D ", "??", " R", "R "))]
    return "yes" if dirty else "no"


def parse_final_metric(log_path: Path) -> tuple[dict[str, str], str, str]:
    if not log_path.exists():
        return {}, "missing_log", ""
    text = log_path.read_text(errors="replace")
    last: dict[str, str] = {}
    for match in METRIC_RE.finditer(text):
        last = match.groupdict()
    if last:
        return last, "ok", ""
    tail = " | ".join(text.splitlines()[-4:])
    tail = tail.replace(str(ROOT) + "/", "")
    if "Traceback" in text or "Error" in text or "ModuleNotFoundError" in text:
        return {}, "failed", tail[:500]
    return {}, "unknown", tail[:500]


def collect_run(run_dir: Path) -> dict[str, Any]:
    meta = read_meta(run_dir / "meta.txt")
    cfg = read_yaml(run_dir / "config.final.yaml") or read_yaml(run_dir / "config.yaml")
    profile_cfg = read_yaml(run_dir / "profile.yaml")
    cfg = deep_update(cfg, profile_cfg)

    exp = cfg.get("experiment", {})
    paths = cfg.get("paths", {})
    train = cfg.get("train", {})
    model = cfg.get("model", {})

    metric, status, notes = parse_final_metric(run_dir / "train.log")

    ckpt = repo_path(paths.get("ckpt"))
    zip_path = repo_path(paths.get("zip"))

    row: dict[str, Any] = {field: "" for field in FIELDS}
    row.update({
        "run_id": run_dir.name,
        "status": status,
        "date": meta.get("date", ""),
        "experiment": exp.get("name", meta.get("config", run_dir.name)),
        "config": meta.get("config", ""),
        "profile": meta.get("profile", ""),
        "python": meta.get("python", "").replace(str(ROOT) + "/", ""),
        "git_dirty": git_dirty_from_meta(run_dir / "meta.txt"),
        "steps_config": train.get("steps", ""),
        "final_step": metric.get("step", ""),
        "final_loss": metric.get("loss", ""),
        "final_offset_mse": metric.get("offset_mse", ""),
        "final_cd": metric.get("cd", ""),
        "final_pred_offset_abs_mean": metric.get("pred_offset_abs_mean", ""),
        "final_pred_offset_l2_mean": metric.get("pred_offset_l2_mean", ""),
        "elapsed_sec": metric.get("elapsed_sec", ""),
        "batch_size": train.get("batch_size", ""),
        "num_points": train.get("num_points", ""),
        "limit": train.get("limit", ""),
        "lr": train.get("lr", ""),
        "cd_weight": train.get("cd_weight", ""),
        "k": model.get("k", ""),
        "feat_dim": model.get("feat_dim", ""),
        "hidden": model.get("hidden", ""),
        "pwsenel": model.get("pwsenel", ""),
        "staas": model.get("staas", ""),
        "staas_strength": model.get("staas_strength", ""),
        "ckpt": display_path(ckpt),
        "ckpt_exists": "yes" if ckpt and ckpt.exists() else "no",
        "zip": display_path(zip_path),
        "zip_exists": "yes" if zip_path and zip_path.exists() else "no",
        "out_dir": display_path(repo_path(paths.get("out_dir"))),
        "notes": notes,
    })
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--print", action="store_true", dest="print_table")
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(f"runs_dir not found: {runs_dir}")

    rows = [collect_run(d) for d in sorted(runs_dir.iterdir()) if d.is_dir()]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {out} rows={len(rows)}")
    if args.print_table:
        writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
