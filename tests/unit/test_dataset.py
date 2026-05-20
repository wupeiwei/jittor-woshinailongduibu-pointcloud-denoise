"""Unit tests for :class:`denoise_baseline.ObjDenoiseDataset`.

Most CI hosts will not have the real ShapeNet dataset, so we use two
strategies:

1. A tiny on-disk OBJ fixture (``tmp_obj_dataset``) for tests that need the
   real ``load_obj_vertices`` -> ``normalize_pc`` -> ``sample_points`` chain.
2. ``unittest.mock.patch`` for tests that want to isolate dataset bookkeeping
   from filesystem / trimesh behavior.

These tests rely only on NumPy and Python stdlib; they do not require Jittor.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# ``denoise_baseline`` imports Jittor and trimesh at top of file, so the whole
# module is unimportable without both. Skip the file if either is missing.
from tests.conftest import HAS_BASELINE_DEPS, HAS_JITTOR, BASELINE_DEPS_ERROR, JITTOR_IMPORT_ERROR

pytestmark = pytest.mark.skipif(
    not HAS_BASELINE_DEPS,
    reason=f"denoise_baseline deps unavailable: {BASELINE_DEPS_ERROR}",
)


def test_dataset_initialization_finds_listed_obj(tmp_obj_dataset):
    """Listed entries that exist on disk must populate ``self.files``."""
    from denoise_baseline import ObjDenoiseDataset

    ds = ObjDenoiseDataset(
        data_root=tmp_obj_dataset["data_root"],
        list_file=tmp_obj_dataset["list_file"],
        num_points=8,
    )

    assert len(ds) == 1
    assert ds.files[0] == tmp_obj_dataset["obj_path"]


def test_dataset_initialization_raises_on_empty_match(tmp_path):
    """If no listed file actually exists, the constructor must error."""
    from denoise_baseline import ObjDenoiseDataset

    list_file = tmp_path / "datalist.txt"
    list_file.write_text("shapenet/missing/category/asset_id\n")

    with pytest.raises(RuntimeError, match="no obj files found"):
        ObjDenoiseDataset(
            data_root=str(tmp_path),
            list_file=str(list_file),
            num_points=8,
        )


def test_dataset_limit_truncates_file_list(tmp_obj_dataset, tmp_path):
    """``limit`` must cap the number of ids parsed from the datalist."""
    from denoise_baseline import ObjDenoiseDataset

    # Append a non-existent id to the list; ``limit=1`` should make us never
    # touch it (so no RuntimeError) and keep only the real OBJ.
    list_file = tmp_path / "limited.txt"
    list_file.write_text(
        tmp_obj_dataset["rel_id"] + "\n" + "shapenet/zzz/never_exists\n"
    )

    ds = ObjDenoiseDataset(
        data_root=tmp_obj_dataset["data_root"],
        list_file=str(list_file),
        num_points=4,
        limit=1,
    )

    assert len(ds) == 1


def test_dataset_getitem_returns_noisy_clean_pair(tmp_obj_dataset):
    """``__getitem__`` must yield matching float32 (N, 3) noisy / clean arrays."""
    from denoise_baseline import ObjDenoiseDataset

    num_points = 16
    ds = ObjDenoiseDataset(
        data_root=tmp_obj_dataset["data_root"],
        list_file=tmp_obj_dataset["list_file"],
        num_points=num_points,
        noise_min=0.005,
        noise_max=0.02,
    )

    noisy, clean = ds[0]

    assert isinstance(noisy, np.ndarray) and isinstance(clean, np.ndarray)
    assert noisy.dtype == np.float32 and clean.dtype == np.float32
    assert noisy.shape == (num_points, 3)
    assert clean.shape == (num_points, 3)
    # Noise is additive Gaussian with sigma > 0, so the two arrays cannot be
    # identical (with overwhelmingly high probability for N=16, 3 channels).
    assert not np.array_equal(noisy, clean)
    # Sanity: residual should be bounded by a few sigma; >0.5 would imply the
    # noise schedule was mis-applied.
    assert np.max(np.abs(noisy - clean)) < 0.5


def test_dataset_cache_clean_reuses_loaded_vertices(tmp_obj_dataset):
    """With ``cache_clean=True`` repeated reads must not re-call the loader."""
    from denoise_baseline import ObjDenoiseDataset

    ds = ObjDenoiseDataset(
        data_root=tmp_obj_dataset["data_root"],
        list_file=tmp_obj_dataset["list_file"],
        num_points=8,
        cache_clean=True,
    )

    # First access populates the cache via the real loader.
    ds[0]
    assert 0 in ds._clean_cache

    # Subsequent access must hit the cache, not re-enter load_obj_vertices.
    with patch("denoise_baseline.load_obj_vertices") as mock_loader:
        noisy, clean = ds[0]
        mock_loader.assert_not_called()

    assert noisy.shape == (8, 3) and clean.shape == (8, 3)


def test_dataset_accepts_short_ids_without_shapenet_prefix(tmp_path):
    """``category/asset_id`` entries should resolve under ``data_root/shapenet/...``."""
    from denoise_baseline import ObjDenoiseDataset

    category, asset = "04401088", "abcdef"
    rel = Path("shapenet") / category / asset / "models" / "model_normalized.obj"
    obj_path = tmp_path / rel
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    obj_path.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
        "f 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n"
    )

    list_file = tmp_path / "short.txt"
    list_file.write_text(f"{category}/{asset}\n")

    ds = ObjDenoiseDataset(
        data_root=str(tmp_path),
        list_file=str(list_file),
        num_points=4,
    )

    assert len(ds) == 1
    assert ds.files[0] == obj_path
