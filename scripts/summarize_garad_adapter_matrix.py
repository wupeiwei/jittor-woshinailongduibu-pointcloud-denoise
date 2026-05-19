#!/usr/bin/env python3
"""Summarize GARA-D adapter matrix runs."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: summarize_garad_adapter_matrix.py <matrix_out_dir>")
    base = Path(sys.argv[1])
    rows = []
    for path in sorted(base.glob("*/*/summary.json")):
        data = json.loads(path.read_text())
        args = data["args"]
        e = data["eval"]
        rows.append({
            "candidate": path.parent.parent.name,
            "run": path.parent.name,
            "model": args["model"],
            "base_k": args["base_k"],
            "base_alpha": args["base_alpha"],
            "lambda_cd": args["lambda_cd"],
            "lambda_delta": args["lambda_delta"],
            "cd_noisy": e["cd_noisy"],
            "cd_base": e["cd_base"],
            "cd_pred": e["cd_pred"],
            "base_better_than_noisy_rate": e["base_better_than_noisy_rate"],
            "pred_better_than_base_rate": e["pred_better_than_base_rate"],
            "score_vs_base": e["score_vs_base"],
            "delta_l2_mean": e["delta_l2_mean"],
            "gate_mean": e["gate_mean"],
            "pass_gate": int(e["cd_pred"] < e["cd_base"] and e["pred_better_than_base_rate"] >= 0.60),
        })
    if not rows:
        raise SystemExit(f"no summary.json found under {base}")

    out_csv = base / "adapter_matrix_summary.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    train_rows = [r for r in rows if r["model"] != "zero"]
    train_rows_sorted = sorted(
        train_rows,
        key=lambda r: (r["pass_gate"], r["pred_better_than_base_rate"], r["cd_base"] - r["cd_pred"]),
        reverse=True,
    )
    lines = [
        "# GARA-D v0.1 noisy-only smoothing adapter matrix",
        "",
        f"Summary CSV: `{out_csv.name}`",
        "",
        "Gate: `cd_pred < cd_base` and `pred_better_than_base_rate >= 0.60`.",
        "",
        "## Runs",
        "",
        "| candidate | run | model | cd_base | cd_pred | win_rate | pass | delta_l2_mean | gate_mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['candidate']} | {r['run']} | {r['model']} | "
            f"{r['cd_base']:.8g} | {r['cd_pred']:.8g} | "
            f"{r['pred_better_than_base_rate']:.3f} | {r['pass_gate']} | "
            f"{r['delta_l2_mean']:.8g} | {r['gate_mean']:.8g} |"
        )
    lines.extend(["", "## Ranking", ""])
    for i, r in enumerate(train_rows_sorted[:10], 1):
        delta = r["cd_base"] - r["cd_pred"]
        lines.append(
            f"{i}. `{r['candidate']}/{r['run']}` model={r['model']} "
            f"pass={r['pass_gate']} win_rate={r['pred_better_than_base_rate']:.3f} "
            f"cd_gain={delta:.8g} cd_base={r['cd_base']:.8g} cd_pred={r['cd_pred']:.8g}"
        )
    lines.extend([
        "",
        "## Decision hint",
        "",
        "If no trained run passes gate, do not sync to A6000; revise correspondence/base/objective again.",
        "If only residual_mlp passes but GARA-D fails, model constraint/gating is too restrictive.",
        "If GARA-D passes on one candidate, repeat with larger eval-limit before any remote train.",
    ])
    report = base / "diagnosis.md"
    report.write_text("\n".join(lines) + "\n")
    print(json.dumps({"summary_csv": str(out_csv), "report": str(report), "rows": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
