#!/usr/bin/env python3
"""Package official starter_code VM outputs into formal submission layout.

The official VM writer may save predictions under paths such as:
  results_quick/dataset_test_noisy/dataset_test_noisy/shapenet/.../denoised.npy
because it joins writer.save_dir with the original asset path.

This script normalizes any output tree containing shapenet/<synset>/<model>/denoised.npy
into a zip whose entries are exactly:
  shapenet/<synset>/<model>/denoised.npy
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def repo_path(text: str) -> Path:
    # wrapper 可从任意 cwd 调用；相对路径统一按仓库根目录解析。
    p = Path(text)
    return p if p.is_absolute() else ROOT / p


def formal_arcname(path: Path, output_root: Path) -> str | None:
    # 官方 writer 可能额外嵌套若干层目录；这里只截取最后的正式 shapenet/.../denoised.npy。
    rel = path.relative_to(output_root)
    parts = rel.parts
    for i, part in enumerate(parts):
        if part == "shapenet" and len(parts) - i == 4 and parts[-1] == "denoised.npy":
            return str(Path(*parts[i:]))
    return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    # zip 可能较大，按块计算 hash，避免一次读入内存。
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    # 本脚本只做“已生成 VM 输出 -> 正式 zip”的包装，不运行 starter_code 推理。
    ap.add_argument("--output-root", required=True, help="official VM output root to scan")
    ap.add_argument("--zip", required=True, help="destination submission zip")
    ap.add_argument("--expected-count", type=int, default=0, help="optional expected denoised.npy count")
    ap.add_argument("--require-float32", action="store_true")
    args = ap.parse_args()

    output_root = repo_path(args.output_root).resolve()
    zip_path = repo_path(args.zip).resolve()
    if not output_root.exists():
        raise FileNotFoundError(f"output root not found: {output_root}")

    entries: dict[str, Path] = {}
    bad: list[str] = []
    for path in sorted(output_root.rglob("denoised.npy")):
        # 逐个文件校验后再写 zip；发现多个问题时一次性报告，方便修路径。
        arc = formal_arcname(path, output_root)
        if arc is None:
            bad.append(f"cannot normalize path: {path}")
            continue
        if arc in entries:
            bad.append(f"duplicate formal entry {arc}: {entries[arc]} and {path}")
            continue
        try:
            arr = np.load(path, allow_pickle=False)
        except Exception as exc:  # noqa: BLE001
            bad.append(f"cannot load {path}: {exc}")
            continue
        if arr.ndim != 2 or arr.shape[-1] != 3:
            bad.append(f"not Nx3: {path} shape={arr.shape}")
        if args.require_float32 and arr.dtype != np.float32:
            bad.append(f"non-float32: {path} dtype={arr.dtype}")
        if not np.isfinite(arr).all():
            bad.append(f"NaN/Inf: {path}")
        entries[arc] = path

    if args.expected_count and len(entries) != args.expected_count:
        bad.append(f"entry count mismatch: got {len(entries)}, expected {args.expected_count}")
    if not entries:
        bad.append(f"no denoised.npy files found under {output_root}")
    if bad:
        print("package official VM outputs FAILED", file=sys.stderr)
        for item in bad[:30]:
            print(f"- {item}", file=sys.stderr)
        if len(bad) > 30:
            print(f"... {len(bad) - 30} more", file=sys.stderr)
        raise SystemExit(1)

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # sorted 保证同一输出树重复打包时 zip 条目顺序稳定，便于 hash/审计。
        for arc, path in sorted(entries.items()):
            zf.write(path, arc)

    print("package official VM outputs OK")
    print(f"output_root: {output_root}")
    print(f"zip: {zip_path}")
    print(f"files: {len(entries)}")
    print(f"sha256: {sha256_file(zip_path)}")


if __name__ == "__main__":
    main()
