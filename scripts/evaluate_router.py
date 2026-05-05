#!/usr/bin/env python3
"""Summarize synthetic denoising CSVs as oracle or rule-based noise routers.

This lightweight script works on outputs produced by scripts/evaluate_cd.py.
It does not re-run models; it answers: if we select a checkpoint by known
noise band or by per-sample best prediction, what is the local upper bound?
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, median


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["_source"] = str(path)
    return rows


def key_row(r: dict[str, str]) -> tuple[str, str]:
    return (r.get("sample", ""), r.get("idx", ""))


def summarize(name: str, rows: list[dict[str, str]]) -> None:
    scores = [float(r["cd_score"]) for r in rows]
    ratios = [float(r["cd_ratio"]) for r in rows]
    better = [float(r["pred_better"]) for r in rows]
    pred = [float(r["cd_pred"]) for r in rows]
    l2 = [float(r["pred_offset_l2_mean"]) for r in rows]
    print(
        f"{name:28s} n={len(rows):4d} "
        f"score={mean(scores):6.2f} median={median(scores):6.2f} "
        f"ratio={mean(ratios):.3f} better={mean(better):.3f} "
        f"pred={mean(pred):.9f} l2={mean(l2):.6f}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--low", action="append", default=[], help="tag=csv for low candidate")
    p.add_argument("--mid", action="append", default=[], help="tag=csv for mid candidate")
    p.add_argument("--high", action="append", default=[], help="tag=csv for high candidate")
    p.add_argument("--out", default="", help="optional selected rows csv")
    args = p.parse_args()

    bands = {"low": args.low, "mid": args.mid, "high": args.high}
    selected_all: list[dict[str, str]] = []
    oracle_all: list[dict[str, str]] = []

    for band, specs in bands.items():
        if not specs:
            continue
        candidates: dict[str, list[dict[str, str]]] = {}
        for spec in specs:
            tag, path = spec.split("=", 1)
            rows = load_rows(Path(path))
            candidates[tag] = rows
            summarize(f"{band}:{tag}", rows)

        # Oracle per-sample across candidates for this band.
        by_key: dict[tuple[str, str], list[tuple[str, dict[str, str]]]] = {}
        for tag, rows in candidates.items():
            for r in rows:
                by_key.setdefault(key_row(r), []).append((tag, r))
        oracle_rows: list[dict[str, str]] = []
        for key, vals in by_key.items():
            # Lower cd_pred is the actual objective for this synthetic set.
            tag, row = min(vals, key=lambda x: float(x[1]["cd_pred"]))
            rr = dict(row)
            rr["router_band"] = band
            rr["router_tag"] = tag
            oracle_rows.append(rr)
        oracle_rows.sort(key=lambda r: int(r.get("idx", "0") or 0))
        summarize(f"{band}:oracle", oracle_rows)
        oracle_all.extend(oracle_rows)

        # Rule-based currently means first candidate in each band.
        first_tag = next(iter(candidates))
        first_rows = []
        for r in candidates[first_tag]:
            rr = dict(r)
            rr["router_band"] = band
            rr["router_tag"] = first_tag
            first_rows.append(rr)
        selected_all.extend(first_rows)
        print()

    if selected_all:
        summarize("rule_router_all", selected_all)
    if oracle_all:
        summarize("oracle_all", oracle_all)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(oracle_all[0].keys()) if oracle_all else []
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(oracle_all)
        print(f"out: {out}")


if __name__ == "__main__":
    main()
