#!/usr/bin/env python3
"""Environment smoke test for Jittor point-cloud denoising.

This script is intentionally diagnostic: it prints enough information to debug
RTX 5060 Ti / RTX A6000 migration issues before a long training run starts.
"""
from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from typing import Iterable


def run(cmd: Iterable[str]) -> str | None:
    cmd = list(cmd)
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        return out.strip()
    except Exception as e:
        print(f"{' '.join(cmd)} failed: {e}")
        return None


def first_line(text: str | None) -> str:
    if not text:
        return ""
    return text.splitlines()[0] if text.splitlines() else text


def print_cmd(cmd: Iterable[str]) -> None:
    cmd = list(cmd)
    print("$", " ".join(cmd))
    out = run(cmd)
    if out:
        print(out)


def require_module(name: str, hint: str) -> None:
    if importlib.util.find_spec(name) is None:
        raise RuntimeError(f"missing Python module: {name}. {hint}")


def main() -> None:
    print("python:", sys.version.replace("\n", " "))
    print("executable:", sys.executable)
    print("platform:", platform.platform())
    print("cwd:", os.getcwd())
    print("CC:", os.environ.get("CC"))
    print("CXX:", os.environ.get("CXX"))
    print("DISABLE_MULTIPROCESSING:", os.environ.get("DISABLE_MULTIPROCESSING"))
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
    print("gcc:", shutil.which("gcc"))
    print("g++:", shutil.which("g++"))
    print("nvcc:", shutil.which("nvcc"))
    print("nvidia-smi:", shutil.which("nvidia-smi"))

    for cmd in [["gcc", "--version"], ["g++", "--version"]]:
        print("$", " ".join(cmd))
        out = run(cmd)
        if out:
            print(first_line(out))

    smi = run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ])
    if smi:
        print("nvidia-smi gpu query:")
        print(smi)
    else:
        print("WARNING: nvidia-smi GPU query failed; CPU-only or driver not visible.")

    # CuPy is not optional for CUDA Jittor training on the target machines.
    require_module(
        "cupy",
        "Install with: $PYTHON -m pip install 'cupy-cuda12x>=13.0,<14.0' "
        "or use PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple bash scripts/install_deps.sh",
    )
    import cupy as cp

    print("cupy:", cp.__version__)
    try:
        print("cupy_cuda_runtime:", cp.cuda.runtime.runtimeGetVersion())
        print("cupy_visible_devices:", cp.cuda.runtime.getDeviceCount())
    except Exception as e:
        print("cupy_cuda_check_failed:", repr(e))
        raise

    try:
        import jittor as jt

        jt.flags.use_cuda = 1
        x = jt.ones((2, 3))
        y = (x * 2).sum()
        print("jittor:", jt.__version__)
        print("jittor_cuda_ok:", x.numpy().shape, "sum=", float(y.data[0]))
    except Exception as e:
        print("jittor_check_failed:", repr(e))
        raise


if __name__ == "__main__":
    main()
