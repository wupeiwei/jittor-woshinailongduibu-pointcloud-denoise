#!/usr/bin/env python3
"""Suggest a training profile from visible NVIDIA GPU names/VRAM.

This does not change configs; it only prints a safe recommendation and a short
command template. Profiles are author workflow presets, not hard requirements.
"""
from __future__ import annotations

import re
import subprocess


def main() -> None:
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ], text=True)
    except Exception as e:
        print("Could not query GPU:", e)
        print("Recommended profile: configs/profiles/local_dev.yaml")
        return

    gpus = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 5:
            continue
        idx, name, mem, driver, compute = parts[:5]
        mem_mib = int(re.findall(r"\d+", mem)[0])
        gpus.append((idx, name, mem_mib, driver, compute))

    print("Visible GPUs:")
    for idx, name, mem, driver, compute in gpus:
        print(f"  [{idx}] {name}, {mem} MiB, driver {driver}, compute {compute}")

    max_mem = max((m for _, _, m, _, _ in gpus), default=0)
    names = " ".join(n.lower() for _, n, _, _, _ in gpus)
    if "a6000" in names or max_mem >= 40000:
        rec = "configs/profiles/a6000.yaml"
        note = "Use single-card A6000 first; only consider multi-GPU after single-card reproducibility is stable."
    elif "5060" in names or max_mem >= 12000:
        rec = "configs/profiles/rtx5060ti.yaml"
        note = "If the 5060 Ti has 8GB VRAM, reduce batch_size or num_points before changing model logic."
    else:
        rec = "configs/profiles/local_dev.yaml"
        note = "Use for smoke/debug only, not final training."

    print("Recommended profile:", rec)
    print("Note:", note)
    print("Train command:")
    print(f"  bash scripts/train.sh configs/denoise_baseline.yaml {rec}")


if __name__ == "__main__":
    main()
