#!/usr/bin/env python3
"""Environment smoke test for point-cloud denoising workflows.

The repository has three practical environment levels:
- pure Python / NumPy utilities;
- Jittor CPU smoke tests;
- Jittor CUDA training/inference.

Keep those checks separate so a NumPy-only artifact validator does not fail just
because CUDA/Jittor compilation is broken on the current machine.
"""
from __future__ import annotations

import argparse
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


def print_basic_env() -> None:
    print("python:", sys.version.replace("\n", " "))
    print("executable:", sys.executable)
    print("platform:", platform.platform())
    print("cwd:", os.getcwd())
    print("JITTOR_HOME:", os.environ.get("JITTOR_HOME", "<unset>"))
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


def check_python_numpy() -> None:
    print("\n[check] python/numpy toolchain")
    modules = [
        ("numpy", "Install project dependencies with: bash scripts/install_deps.sh"),
        ("yaml", "PyYAML is required. Install with: python -m pip install PyYAML"),
        ("trimesh", "trimesh is required for OBJ training data loading."),
        ("scipy", "scipy is required by candidate/probe utilities."),
    ]
    for name, hint in modules:
        require_module(name, hint)
        mod = __import__(name)
        print(f"{name}: {getattr(mod, '__version__', 'available')}")
    print("python_numpy_ok: true")


def check_cupy_cuda(require_device: bool) -> bool:
    print("\n[check] cupy/cuda runtime")
    require_module(
        "cupy",
        "Install with: python -m pip install 'cupy-cuda12x>=13.0,<14.0' "
        "or use PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple bash scripts/install_deps.sh",
    )
    import cupy as cp

    print("cupy:", cp.__version__)
    try:
        print("cupy_cuda_runtime:", cp.cuda.runtime.runtimeGetVersion())
        device_count = cp.cuda.runtime.getDeviceCount()
        print("cupy_visible_devices:", device_count)
    except Exception as e:
        print("cupy_cuda_check_failed:", repr(e))
        if require_device:
            raise RuntimeError(
                "CUDA runtime is not usable by CuPy. For CUDA training/inference, "
                "check driver visibility, CUDA_VISIBLE_DEVICES, and the matching cupy-cuda wheel."
            ) from e
        return False
    if require_device and device_count <= 0:
        raise RuntimeError("CUDA mode requested but CuPy reports zero visible devices.")
    return device_count > 0


def check_jittor(use_cuda: bool) -> None:
    label = "jittor_cuda" if use_cuda else "jittor_cpu"
    print(f"\n[check] {label}")
    try:
        import jittor as jt

        jt.flags.use_cuda = 1 if use_cuda else 0
        x = jt.ones((2, 3))
        y = (x * 2).sum()
        print("jittor:", getattr(jt, "__version__", "unknown"))
        print("jittor_file:", getattr(jt, "__file__", "unknown"))
        print(f"{label}_ok:", x.numpy().shape, "sum=", float(y.data[0]))
    except Exception as e:
        print(f"{label}_check_failed:", repr(e))
        if isinstance(e, ModuleNotFoundError) and e.name == "jittor":
            print(
                "hint: Jittor is not installed in the active Python environment. "
                "Run `source scripts/env.sh` and install project dependencies into that environment."
            )
        if "Read-only file system" in repr(e) and ".cache/jittor" in repr(e):
            print(
                "hint: Jittor needs a writable HOME/cache path. Run `source scripts/env.sh` "
                "and, if the current HOME is read-only, point HOME to a writable directory "
                "before importing Jittor."
            )
        if use_cuda:
            print("hint: CUDA mode also requires visible GPU, compatible driver, compiler, and CuPy.")
        raise RuntimeError(f"{label} check failed") from e


def main() -> None:
    p = argparse.ArgumentParser(description="Check project runtime environment by level.")
    p.add_argument(
        "--level",
        choices=["python", "jittor-cpu", "jittor-cuda", "all"],
        default="jittor-cuda",
        help="Default is jittor-cuda because training/prediction wrappers need CUDA.",
    )
    args = p.parse_args()

    print_basic_env()

    try:
        if args.level in {"python", "all"}:
            check_python_numpy()
        if args.level in {"jittor-cpu", "all"}:
            check_jittor(use_cuda=False)
        if args.level in {"jittor-cuda", "all"}:
            check_cupy_cuda(require_device=True)
            check_jittor(use_cuda=True)
    except Exception as e:
        print("\nenvironment check FAILED:", e, file=sys.stderr)
        raise SystemExit(1) from e

    print("\nenvironment check OK:", args.level)


if __name__ == "__main__":
    main()
