#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
from pathlib import Path


def read_routes(path: Path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize UnifiedDenoisePipeline router stress routes.csv files.")
    ap.add_argument("paths", nargs="+", help="routes.csv paths or result directories containing routes.csv")
    args = ap.parse_args()
    for raw in args.paths:
        p = Path(raw)
        if p.is_dir():
            p = p / "routes.csv"
        rows = read_routes(p)
        counts = {}
        gates = []
        stats = []
        hard_counts = {}
        elapsed = 0.0
        for r in rows:
            route = r.get("route", "")
            hard = r.get("hard_route", "")
            counts[route] = counts.get(route, 0) + 1
            hard_counts[hard] = hard_counts.get(hard, 0) + 1
            if r.get("gate") not in (None, ""):
                gates.append(float(r["gate"]))
            if r.get("plane_res_p75") not in (None, ""):
                stats.append(float(r["plane_res_p75"]))
            if r.get("elapsed_sec") not in (None, ""):
                elapsed += float(r["elapsed_sec"])
        print(f"== {p} ==")
        print(f"files={len(rows)} route_counts={counts} hard_counts={hard_counts} elapsed_sec={elapsed:.2f}")
        if stats:
            print(f"plane_res_p75 min={min(stats):.6f} mean={sum(stats)/len(stats):.6f} max={max(stats):.6f}")
        if gates:
            print(f"gate min={min(gates):.4f} mean={sum(gates)/len(gates):.4f} max={max(gates):.4f}")


if __name__ == "__main__":
    main()
