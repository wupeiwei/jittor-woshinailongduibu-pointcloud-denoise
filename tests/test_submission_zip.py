"""Tests for submission zip package format validation.

These tests verify that the competition submission zip meets the official format:
- zip exists and is readable
- entries follow shapenet/<category>/<model>/denoised.npy
- expected file count matches
- every npy has expected shape, numeric dtype, and finite values

Tests that require an actual submission zip file will be skipped if the file
is not present. Mock-based tests verify the format logic independently.
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Constants matching scripts/check_submission.py validation rules
# ---------------------------------------------------------------------------

EXPECTED_COUNT = 200
EXPECTED_SHAPE = (50000, 3)
EXPECTED_PATH_DEPTH = 4  # shapenet/<category>/<model>/denoised.npy
EXPECTED_PREFIX = "shapenet"
EXPECTED_FILENAME = "denoised.npy"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _find_submission_zip() -> Path | None:
    """Try to locate a real submission zip file."""
    # Check environment variable first
    env_zip = os.environ.get("SUBMISSION_ZIP", "")
    if env_zip and Path(env_zip).exists():
        return Path(env_zip)

    # Check common locations in the project
    candidates = [
        PROJECT_ROOT / "result_denoise_baseline.zip",
        PROJECT_ROOT / "result_denoise_baseline_a6000_full.zip",
        PROJECT_ROOT / "result_official_vm_fixed_stitch.zip",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


@pytest.fixture()
def mock_submission_zip(tmp_path) -> Path:
    """Create a valid mock submission zip for format testing."""
    zip_path = tmp_path / "mock_submission.zip"
    rng = np.random.default_rng(seed=42)
    categories = ["02691156", "04379243", "03001627", "04256520"]
    models_per_cat = EXPECTED_COUNT // len(categories)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        count = 0
        for cat in categories:
            for i in range(models_per_cat):
                model_id = f"{cat}_{i:04d}_abcdef1234567890"
                arcname = f"shapenet/{cat}/{model_id}/denoised.npy"
                arr = rng.standard_normal(EXPECTED_SHAPE).astype(np.float32)
                buf = io.BytesIO()
                np.save(buf, arr)
                zf.writestr(arcname, buf.getvalue())
                count += 1
    return zip_path


@pytest.fixture()
def invalid_zip_bad_path(tmp_path) -> Path:
    """Create a zip with invalid entry paths."""
    zip_path = tmp_path / "bad_path.zip"
    rng = np.random.default_rng(seed=42)
    with zipfile.ZipFile(zip_path, "w") as zf:
        # Wrong prefix
        arr = rng.standard_normal(EXPECTED_SHAPE).astype(np.float32)
        buf = io.BytesIO()
        np.save(buf, arr)
        zf.writestr("wrong_prefix/02691156/model001/denoised.npy", buf.getvalue())
        # Missing level
        zf.writestr("shapenet/02691156/denoised.npy", buf.getvalue())
        # Wrong filename
        zf.writestr("shapenet/02691156/model002/result.npy", buf.getvalue())
    return zip_path


@pytest.fixture()
def invalid_zip_bad_shape(tmp_path) -> Path:
    """Create a zip with wrong array shape."""
    zip_path = tmp_path / "bad_shape.zip"
    rng = np.random.default_rng(seed=42)
    with zipfile.ZipFile(zip_path, "w") as zf:
        arr = rng.standard_normal((1000, 3)).astype(np.float32)
        buf = io.BytesIO()
        np.save(buf, arr)
        zf.writestr("shapenet/02691156/model001/denoised.npy", buf.getvalue())
    return zip_path


@pytest.fixture()
def invalid_zip_nan(tmp_path) -> Path:
    """Create a zip with NaN values."""
    zip_path = tmp_path / "nan_values.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        arr = np.full(EXPECTED_SHAPE, np.nan, dtype=np.float32)
        buf = io.BytesIO()
        np.save(buf, arr)
        zf.writestr("shapenet/02691156/model001/denoised.npy", buf.getvalue())
    return zip_path


# ---------------------------------------------------------------------------
# Tests using mock data (always runnable)
# ---------------------------------------------------------------------------


@pytest.mark.submission
class TestSubmissionZipFormat:
    """Test submission zip format rules using mock data."""

    def test_valid_zip_readable(self, mock_submission_zip: Path):
        """Valid zip should be readable without errors."""
        assert mock_submission_zip.exists()
        with zipfile.ZipFile(mock_submission_zip, "r") as zf:
            assert zf.testzip() is None, "zip has corrupt entries"

    def test_valid_zip_file_count(self, mock_submission_zip: Path):
        """Mock zip should have exactly EXPECTED_COUNT files."""
        with zipfile.ZipFile(mock_submission_zip, "r") as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            assert len(names) == EXPECTED_COUNT, (
                f"Expected {EXPECTED_COUNT} files, got {len(names)}"
            )

    def test_valid_zip_naming_convention(self, mock_submission_zip: Path):
        """All entries must follow shapenet/<category>/<model>/denoised.npy."""
        with zipfile.ZipFile(mock_submission_zip, "r") as zf:
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                parts = Path(name).parts
                assert len(parts) == EXPECTED_PATH_DEPTH, (
                    f"Path depth mismatch: {name} has {len(parts)} parts, "
                    f"expected {EXPECTED_PATH_DEPTH}"
                )
                assert parts[0] == EXPECTED_PREFIX, (
                    f"Path must start with '{EXPECTED_PREFIX}', got: {name}"
                )
                assert parts[-1] == EXPECTED_FILENAME, (
                    f"File must be named '{EXPECTED_FILENAME}', got: {name}"
                )

    def test_valid_zip_array_shape(self, mock_submission_zip: Path):
        """Each npy file must have shape (50000, 3)."""
        with zipfile.ZipFile(mock_submission_zip, "r") as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            # Check a subset for performance
            for name in names[:5]:
                with zf.open(name) as f:
                    arr = np.load(io.BytesIO(f.read()), allow_pickle=False)
                assert arr.shape == EXPECTED_SHAPE, (
                    f"{name}: expected shape {EXPECTED_SHAPE}, got {arr.shape}"
                )

    def test_valid_zip_array_dtype(self, mock_submission_zip: Path):
        """Each npy file must have a numeric (float) dtype."""
        with zipfile.ZipFile(mock_submission_zip, "r") as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            for name in names[:5]:
                with zf.open(name) as f:
                    arr = np.load(io.BytesIO(f.read()), allow_pickle=False)
                assert np.issubdtype(arr.dtype, np.floating), (
                    f"{name}: expected float dtype, got {arr.dtype}"
                )

    def test_valid_zip_finite_values(self, mock_submission_zip: Path):
        """Each npy file must contain only finite values (no NaN/Inf)."""
        with zipfile.ZipFile(mock_submission_zip, "r") as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            for name in names[:5]:
                with zf.open(name) as f:
                    arr = np.load(io.BytesIO(f.read()), allow_pickle=False)
                assert np.isfinite(arr).all(), (
                    f"{name}: contains NaN or Inf values"
                )

    def test_no_duplicate_entries(self, mock_submission_zip: Path):
        """Zip must not contain duplicate paths."""
        with zipfile.ZipFile(mock_submission_zip, "r") as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            assert len(names) == len(set(names)), (
                f"Zip contains duplicate entries: {len(names) - len(set(names))} duplicates"
            )


@pytest.mark.submission
class TestSubmissionZipInvalid:
    """Test that invalid zips are correctly detected."""

    def test_bad_path_detection(self, invalid_zip_bad_path: Path):
        """Entries with wrong path structure should be detectable."""
        bad = []
        with zipfile.ZipFile(invalid_zip_bad_path, "r") as zf:
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                parts = Path(name).parts
                if (
                    len(parts) != EXPECTED_PATH_DEPTH
                    or parts[0] != EXPECTED_PREFIX
                    or parts[-1] != EXPECTED_FILENAME
                ):
                    bad.append(name)
        assert len(bad) > 0, "Should detect invalid path entries"

    def test_bad_shape_detection(self, invalid_zip_bad_shape: Path):
        """Arrays with wrong shape should be detectable."""
        with zipfile.ZipFile(invalid_zip_bad_shape, "r") as zf:
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                with zf.open(name) as f:
                    arr = np.load(io.BytesIO(f.read()), allow_pickle=False)
                assert arr.shape != EXPECTED_SHAPE, (
                    "Test fixture should have wrong shape"
                )

    def test_nan_detection(self, invalid_zip_nan: Path):
        """NaN values should be detectable."""
        with zipfile.ZipFile(invalid_zip_nan, "r") as zf:
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                with zf.open(name) as f:
                    arr = np.load(io.BytesIO(f.read()), allow_pickle=False)
                assert not np.isfinite(arr).all(), (
                    "Test fixture should contain NaN"
                )


# ---------------------------------------------------------------------------
# Tests using real submission zip (skipped if not available)
# ---------------------------------------------------------------------------


@pytest.mark.submission
class TestRealSubmissionZip:
    """Tests against a real submission zip. Skipped if no zip is available."""

    @pytest.fixture(autouse=True)
    def _require_zip(self):
        zip_path = _find_submission_zip()
        if zip_path is None:
            pytest.skip(
                "No submission zip found. Set SUBMISSION_ZIP env var or place "
                "a zip in the project root."
            )
        self.zip_path = zip_path

    def test_zip_exists_and_readable(self):
        """Real zip should exist and be readable."""
        assert self.zip_path.exists()
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            assert zf.testzip() is None

    def test_real_zip_file_count(self):
        """Real zip should contain exactly 200 files."""
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            assert len(names) == EXPECTED_COUNT, (
                f"Expected {EXPECTED_COUNT} files, got {len(names)}"
            )

    def test_real_zip_naming(self):
        """Real zip entries must follow the naming convention."""
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                parts = Path(name).parts
                assert len(parts) == EXPECTED_PATH_DEPTH, (
                    f"Bad path: {name}"
                )
                assert parts[0] == EXPECTED_PREFIX
                assert parts[-1] == EXPECTED_FILENAME

    def test_real_zip_point_cloud_format(self):
        """Spot-check a few entries from the real zip for shape/dtype/finite."""
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            for name in names[:3]:
                with zf.open(name) as f:
                    arr = np.load(io.BytesIO(f.read()), allow_pickle=False)
                assert arr.shape == EXPECTED_SHAPE, (
                    f"{name}: got shape {arr.shape}"
                )
                assert np.issubdtype(arr.dtype, np.floating), (
                    f"{name}: got dtype {arr.dtype}"
                )
                assert np.isfinite(arr).all(), (
                    f"{name}: contains NaN/Inf"
                )
