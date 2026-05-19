#!/usr/bin/env python3
"""Validate a point-cloud denoising submission zip.

Checks the formal-competition submission package without relying on the
competition server:
- zip exists and is readable;
- entries follow shapenet/<category>/<model>/denoised.npy;
- expected file count matches;
- every npy has expected shape, numeric dtype, and finite values;
- optionally matches entries against a test_root containing noisy.npy files.
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]


def deep_update(base: dict, patch: dict | None) -> dict:
    # config + profile 合并规则与训练/推理脚本保持一致：profile 只覆盖声明字段。
    for key, value in (patch or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def repo_path(path: str | Path | None) -> Path | None:
    if path is None or str(path) == "":
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_config(config: str, profiles: list[str]) -> dict:
    cfg: dict = {}
    if config:
        cfg = yaml.safe_load(Path(config).read_text()) or {}
    for profile in profiles:
        patch = yaml.safe_load(Path(profile).read_text()) or {}
        cfg = deep_update(cfg, patch)
    return cfg


def expected_entries_from_test_root(test_root: Path) -> set[str]:
    # 用 noisy.npy 目录结构推导提交 zip 必须包含的 denoised.npy 相对路径。
    files = sorted(test_root.glob("shapenet/*/*/noisy.npy"))
    return {str(f.relative_to(test_root).parent / "denoised.npy") for f in files}


def parse_shape(text: str) -> tuple[int, int]:
    parts = [p.strip() for p in text.replace("x", ",").split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("shape must look like 50000,3")
    return int(parts[0]), int(parts[1])


def validate_zip(
    zip_path: Path,
    expected_count: int,
    expected_shape: tuple[int, int],
    test_root: Path | None = None,
    require_float32: bool = False,
) -> None:
    # 这是正式提交前的本地硬门槛：结构、数量、shape、dtype、有限值必须先过。
    if not zip_path.exists():
        raise FileNotFoundError(f"zip not found: {zip_path}")

    # 如果提供 test_root，则进一步校验 zip 条目与隐藏测试 noisy 树一一对应。
    expected_entries = expected_entries_from_test_root(test_root) if test_root else None
    names_seen: set[str] = set()
    bad: list[str] = []
    shapes: dict[tuple[int, ...], int] = {}
    dtypes: dict[str, int] = {}
    finite_min = np.inf
    finite_max = -np.inf

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [name for name in zf.namelist() if not name.endswith("/")]
        # 文件数先做粗筛；后面仍逐项检查路径和 npy 内容。
        if len(names) != expected_count:
            bad.append(f"file count mismatch: got {len(names)}, expected {expected_count}")

        duplicates = len(names) - len(set(names))
        if duplicates:
            bad.append(f"duplicate entries: {duplicates}")

        for name in names:
            names_seen.add(name)
            parts = Path(name).parts
            # 官方格式固定为 shapenet/<category>/<model>/denoised.npy。
            if len(parts) != 4 or parts[0] != "shapenet" or parts[-1] != "denoised.npy":
                bad.append(f"bad path: {name}")
                continue

            try:
                with zf.open(name, "r") as f:
                    arr = np.load(io.BytesIO(f.read()), allow_pickle=False)
            except Exception as exc:  # noqa: BLE001 - report all broken entries together.
                bad.append(f"cannot load {name}: {exc}")
                continue

            shapes[arr.shape] = shapes.get(arr.shape, 0) + 1
            dtypes[str(arr.dtype)] = dtypes.get(str(arr.dtype), 0) + 1

            # 逐项收集所有错误，最后统一输出，避免修一个文件再跑一次。
            if arr.shape != expected_shape:
                bad.append(f"bad shape: {name} got {arr.shape}, expected {expected_shape}")
            if arr.ndim != 2 or arr.shape[-1] != 3:
                bad.append(f"not Nx3: {name} shape={arr.shape}")
            if not np.issubdtype(arr.dtype, np.number):
                bad.append(f"non-numeric dtype: {name} dtype={arr.dtype}")
            if require_float32 and arr.dtype != np.float32:
                bad.append(f"non-float32 dtype: {name} dtype={arr.dtype}")
            if not np.isfinite(arr).all():
                bad.append(f"NaN/Inf found: {name}")
            else:
                finite_min = min(finite_min, float(arr.min()))
                finite_max = max(finite_max, float(arr.max()))

    if expected_entries is not None:
        # test_root match 能抓住漏样本、额外样本和目录层级错误。
        missing = sorted(expected_entries - names_seen)
        extra = sorted(names_seen - expected_entries)
        if missing:
            bad.append(f"missing entries vs test_root: {len(missing)} first={missing[:5]}")
        if extra:
            bad.append(f"extra entries vs test_root: {len(extra)} first={extra[:5]}")

    if bad:
        print("submission check FAILED", file=sys.stderr)
        for item in bad[:30]:
            print(f"- {item}", file=sys.stderr)
        if len(bad) > 30:
            print(f"... {len(bad) - 30} more", file=sys.stderr)
        raise SystemExit(1)

    print("submission check OK")
    print(f"zip: {zip_path}")
    print(f"files: {len(names_seen)}")
    print(f"shapes: {shapes}")
    print(f"dtypes: {dtypes}")
    print(f"finite_range: [{finite_min:.6f}, {finite_max:.6f}]")
    if expected_entries is not None:
        print(f"matched test_root entries: {len(expected_entries)}")


def main() -> None:
    p = argparse.ArgumentParser()
    # zip 可从 CLI 直接传入，也可从 config paths.zip 读取，方便脚本链复用。
    p.add_argument("zip", nargs="?", default="", help="submission zip path; defaults to config paths.zip")
    p.add_argument("--config", default="configs/denoise_baseline.yaml")
    p.add_argument("--profile", action="append", default=[])
    p.add_argument("--test-root", default="", help="optional test root override; defaults to config paths.test_root")
    p.add_argument("--expected-count", type=int, default=200)
    p.add_argument("--expected-shape", type=parse_shape, default=(50000, 3))
    p.add_argument("--require-float32", action="store_true")
    p.add_argument("--no-test-root-match", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config, args.profile)
    paths = cfg.get("paths", {})

    zip_path = repo_path(args.zip or paths.get("zip", "result_denoise_baseline.zip"))
    test_root = None
    if not args.no_test_root_match:
        test_root = repo_path(args.test_root or paths.get("test_root", ""))
        if test_root is not None and not test_root.exists():
            raise FileNotFoundError(f"test_root not found: {test_root}")

    validate_zip(
        zip_path=zip_path,
        expected_count=args.expected_count,
        expected_shape=args.expected_shape,
        test_root=test_root,
        require_float32=args.require_float32,
    )


if __name__ == "__main__":
    main()
