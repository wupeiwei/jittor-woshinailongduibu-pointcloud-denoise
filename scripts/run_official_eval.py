#!/usr/bin/env python3
"""Wrapper around starter_code/evaluate.py for Candidate Registry usage.

The official evaluator prints human-readable CD/P2S scores. This wrapper keeps
that entry point intact, captures its stdout, extracts the key numbers when
possible, and writes a JSON sidecar that `scripts/candidate_registry.py` can
record. It does not run prediction or training.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def repo_path(path: str | Path | None) -> Path | None:
    # 允许 registry/脚本链传仓库相对路径；外部绝对路径也原样保留。
    if path is None or str(path) == "":
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def extract_score(patterns: list[str], text: str) -> float | None:
    # 官方 evaluate.py 输出是面向人读的文本，这里用多组正则兼容中文/英文字段名。
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def parse_scores(stdout: str) -> dict[str, float | None]:
    cd = extract_score([
        r"CD\s*得分:\s*([0-9.]+)",
        r"CD_score\s*[=:]\s*([0-9.]+)",
    ], stdout)
    p2s = extract_score([
        r"P2S\s*得分:\s*([0-9.]+)",
        r"P2S_score\s*[=:]\s*([0-9.]+)",
    ], stdout)
    final = extract_score([
        r"最终得分[^:]*:\s*([0-9.]+)",
        r"final_score\s*[=:]\s*([0-9.]+)",
    ], stdout)
    return {"cd_score": cd, "p2s_score": p2s, "final_score": final}


def main() -> None:
    p = argparse.ArgumentParser(description="Run official starter_code/evaluate.py and save JSON summary.")
    # 只包装官方评估，不改变 evaluate.py 参数语义；额外参数通过 --extra 透传。
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--gt-dir", required=True)
    p.add_argument("--noisy-dir", required=True)
    p.add_argument("--mesh-dir", default="")
    p.add_argument("--out-json", default="experiments/official_eval.json")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--extra", action="append", default=[], help="Extra raw argument passed to starter_code/evaluate.py")
    args = p.parse_args()

    evaluator = ROOT / "starter_code" / "evaluate.py"
    out_json = repo_path(args.out_json)
    assert out_json is not None
    cmd = [
        args.python,
        str(evaluator),
        "--pred_dir",
        str(repo_path(args.pred_dir) or args.pred_dir),
        "--gt_dir",
        str(repo_path(args.gt_dir) or args.gt_dir),
        "--noisy_dir",
        str(repo_path(args.noisy_dir) or args.noisy_dir),
        "--workers",
        str(args.workers),
    ]
    if args.mesh_dir:
        cmd.extend(["--mesh_dir", str(repo_path(args.mesh_dir) or args.mesh_dir)])
    if args.verbose:
        cmd.append("--verbose")
    cmd.extend(args.extra)

    # 捕获 stdout/stderr 并原样回显，同时写入 JSON sidecar 供 Candidate Registry 记录。
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    scores = parse_scores(proc.stdout)
    payload = {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "command": cmd,
        "returncode": proc.returncode,
        "evaluate_py": rel(evaluator),
        "pred_dir": args.pred_dir,
        "gt_dir": args.gt_dir,
        "noisy_dir": args.noisy_dir,
        "mesh_dir": args.mesh_dir,
        **scores,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"official eval json: {rel(out_json)}")
    if proc.returncode != 0:
        # 官方评估失败时保留 JSON/日志，但把失败码继续传给上层脚本。
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
