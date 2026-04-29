#!/usr/bin/env python3
"""Check formal-track dataset layout before training or prediction."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def resolve_repo_path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def deep_update(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in (patch or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config_paths(config: str, profiles: list[str]) -> dict[str, str]:
    cfg: dict[str, Any] = {}
    if config:
        cfg = yaml.safe_load(resolve_repo_path(config).read_text()) or {}
    for profile in profiles:
        patch = yaml.safe_load(resolve_repo_path(profile).read_text()) or {}
        cfg = deep_update(cfg, patch)
    paths = cfg.get("paths", {}) or {}
    return {
        "data_root": paths.get("data_root", "dataset_train"),
        "test_root": paths.get("test_root", "dataset_test_noisy"),
        "train_list": paths.get("train_list", "starter_code/datalist/train.txt"),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="", help="Optional YAML config; paths.* values are checked")
    p.add_argument("--profile", action="append", default=[], help="Optional profile override; can be repeated")
    p.add_argument("--data-root", default=None)
    p.add_argument("--test-root", default=None)
    p.add_argument("--train-list", default=None)
    p.add_argument("--limit", type=int, default=5)
    args = p.parse_args()

    cfg_paths = load_config_paths(args.config, args.profile)
    data_root = resolve_repo_path(args.data_root or cfg_paths["data_root"])
    test_root = resolve_repo_path(args.test_root or cfg_paths["test_root"])
    train_list = resolve_repo_path(args.train_list or cfg_paths["train_list"])
    print("data_root:", data_root.resolve(), "exists=", data_root.exists())
    print("test_root:", test_root.resolve(), "exists=", test_root.exists())
    print("train_list:", train_list.resolve(), "exists=", train_list.exists())

    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")
    if not test_root.exists():
        raise FileNotFoundError(f"test_root not found: {test_root}")
    if not train_list.exists():
        raise FileNotFoundError(f"train_list not found: {train_list}")

    ids = [x.strip() for x in train_list.read_text().splitlines() if x.strip()]
    print("train_list entries:", len(ids))
    ok = 0
    missing = []
    sample_n = min(len(ids), max(args.limit, 0))
    for x in ids[:sample_n]:
        rel = Path(x)
        if rel.parts and rel.parts[0] == "shapenet":
            f = data_root / rel / "models" / "model_normalized.obj"
        else:
            f = data_root / "shapenet" / rel / "models" / "model_normalized.obj"
        if f.exists():
            ok += 1
        else:
            missing.append(str(f))
    print(f"train obj sample ok: {ok}/{sample_n}")
    if missing:
        print("missing samples:")
        for m in missing[:10]:
            print(" ", m)
        raise FileNotFoundError(f"missing {len(missing)} sampled training objects")

    noisy = sorted(test_root.glob("shapenet/*/*/noisy.npy"))
    print("test noisy.npy count:", len(noisy))
    if not noisy:
        raise FileNotFoundError(f"no noisy.npy files found under {test_root}/shapenet")
    for f in noisy[: args.limit]:
        arr = np.load(f, mmap_mode="r")
        print(" ", f.relative_to(ROOT) if f.is_relative_to(ROOT) else f, arr.shape, arr.dtype)


if __name__ == "__main__":
    main()
