"""Pytest fixtures and import-time setup for the Jittor denoising test suite.

This conftest is intentionally defensive: the project depends on Jittor, but
many development / CI machines do not have a working CUDA toolkit. To let the
pure-Python utility tests run on CPU-only hosts, we clear ``nvcc_path`` before
Jittor's first import so the framework falls back to its CPU backend instead
of failing at ``check_cuda``.

If Jittor still fails to import (e.g. compilation toolchain missing), the
module-level ``HAS_JITTOR`` flag stays ``False`` and individual test files use
``pytest.importorskip`` / ``pytest.mark.skipif`` to skip gracefully.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# --- sys.path setup so `import denoise_baseline` / `import denoise_utils` work
# regardless of the directory pytest is launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Force Jittor into CPU mode before it gets imported anywhere. The empty
# ``nvcc_path`` env var makes ``jittor.compiler.check_cuda`` return early
# instead of trying to dlopen a (possibly missing) libcudart.so.
os.environ.setdefault("nvcc_path", "")

# Best-effort Jittor probe. Tests that need Jittor key off ``HAS_JITTOR``;
# tests that only need NumPy still run when Jittor is absent.
try:  # pragma: no cover - environment dependent
    import jittor as jt  # noqa: F401

    jt.flags.use_cuda = 0
    HAS_JITTOR = True
    JITTOR_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - environment dependent
    HAS_JITTOR = False
    JITTOR_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

# Best-effort trimesh probe. ``denoise_baseline`` imports trimesh at module
# level, so tests importing from it also need trimesh available.
try:  # pragma: no cover - environment dependent
    import trimesh  # noqa: F401

    HAS_TRIMESH = True
    TRIMESH_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - environment dependent
    HAS_TRIMESH = False
    TRIMESH_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

# Combined flag: tests that import denoise_baseline need both Jittor and trimesh.
HAS_BASELINE_DEPS = HAS_JITTOR and HAS_TRIMESH
BASELINE_DEPS_ERROR = JITTOR_IMPORT_ERROR or TRIMESH_IMPORT_ERROR


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to the repository root."""
    return PROJECT_ROOT


@pytest.fixture()
def rng() -> np.random.Generator:
    """Deterministic NumPy RNG so tests stay reproducible across runs."""
    return np.random.default_rng(seed=20260520)


@pytest.fixture()
def random_pointcloud(rng):
    """Factory producing random ``(B, N, C)`` point clouds as NumPy arrays.

    The default returns a small, CPU-friendly cloud suitable for unit tests
    that exercise tensor shapes without stressing the JIT compiler.
    """

    def _make(batch: int = 2, num_points: int = 64, channels: int = 3) -> np.ndarray:
        return rng.standard_normal((batch, num_points, channels)).astype(np.float32)

    return _make


@pytest.fixture()
def random_knn_idx(rng):
    """Factory producing random KNN-style index tensors of shape ``(B, N, K)``."""

    def _make(batch: int = 2, num_points: int = 64, k: int = 8) -> np.ndarray:
        return rng.integers(0, num_points, size=(batch, num_points, k)).astype(np.int32)

    return _make


@pytest.fixture()
def tmp_obj_dataset(tmp_path):
    """Build a tiny on-disk ShapeNet-like layout with a single .obj file.

    Returns a dict with ``data_root``, ``list_file`` (path) and ``rel_id`` so
    tests can instantiate :class:`ObjDenoiseDataset` against real files
    without needing the full competition dataset.
    """
    rel_id = "shapenet/00000000/00000000abcdef"
    obj_dir = tmp_path / rel_id / "models"
    obj_dir.mkdir(parents=True, exist_ok=True)

    # Minimal valid OBJ: a unit tetrahedron. trimesh can load this with
    # ``force="mesh"`` and produces 4 vertices, which is enough to exercise
    # the sampling / noise injection branches of ObjDenoiseDataset.
    obj_path = obj_dir / "model_normalized.obj"
    obj_path.write_text(
        "v 0.0 0.0 0.0\n"
        "v 1.0 0.0 0.0\n"
        "v 0.0 1.0 0.0\n"
        "v 0.0 0.0 1.0\n"
        "f 1 2 3\n"
        "f 1 2 4\n"
        "f 1 3 4\n"
        "f 2 3 4\n"
    )

    list_file = tmp_path / "datalist.txt"
    list_file.write_text(rel_id + "\n")

    return {
        "data_root": str(tmp_path),
        "list_file": str(list_file),
        "rel_id": rel_id,
        "obj_path": obj_path,
    }
