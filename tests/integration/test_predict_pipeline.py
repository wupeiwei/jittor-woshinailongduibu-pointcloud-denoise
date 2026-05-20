"""Integration tests for the end-to-end prediction path.

These tests instantiate :class:`denoise_baseline.ResidualDenoiser` and run a
forward pass on a mock point cloud (no real ShapeNet files needed). They are
marked with the ``gpu`` marker because the full encoder + heads compile a
large CUDA kernel set on first use; CPU-only environments are advised to
deselect this file via ``-m "not gpu"``.

If Jittor is unavailable, the whole file is skipped via ``pytestmark``.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import HAS_JITTOR, JITTOR_IMPORT_ERROR

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not HAS_JITTOR,
        reason=f"Jittor unavailable in this environment: {JITTOR_IMPORT_ERROR}",
    ),
]


def _make_noisy_cloud(batch: int = 2, num_points: int = 64, sigma: float = 0.01):
    """Produce a synthetic noisy cloud sampled from the unit sphere surface."""
    rng = np.random.default_rng(20260520)
    direction = rng.standard_normal((batch, num_points, 3)).astype(np.float32)
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True) + 1e-9
    noise = rng.normal(0.0, sigma, direction.shape).astype(np.float32)
    return direction + noise


def test_residual_denoiser_baseline_forward_shape():
    """Pure baseline (no extras) keeps the input (B, N, 3) shape."""
    import jittor as jt

    from denoise_baseline import ResidualDenoiser

    cloud = _make_noisy_cloud()
    model = ResidualDenoiser(k=8, feat_dim=32, hidden=32)
    pred = model(jt.array(cloud))

    assert tuple(pred.shape) == cloud.shape


def test_residual_denoiser_return_offset_pair():
    """``return_offset=True`` returns (pred, offset) with matching shapes."""
    import jittor as jt

    from denoise_baseline import ResidualDenoiser

    cloud = _make_noisy_cloud()
    model = ResidualDenoiser(k=8, feat_dim=32, hidden=32)
    result = model(jt.array(cloud), return_offset=True)

    # The model may return a tuple or a single tensor depending on flags;
    # baseline configuration must return the (pred, offset) pair.
    assert isinstance(result, tuple) and len(result) == 2
    pred, offset = result
    assert tuple(pred.shape) == cloud.shape
    assert tuple(offset.shape) == cloud.shape


def test_residual_denoiser_with_pwsenel_and_move_gate():
    """Enabling PWSENEL + move_gate keeps the forward path valid."""
    import jittor as jt

    from denoise_baseline import ResidualDenoiser

    cloud = _make_noisy_cloud()
    model = ResidualDenoiser(
        k=8,
        feat_dim=32,
        hidden=32,
        use_pwsenel=True,
        use_move_gate=True,
    )
    pred = model(jt.array(cloud))

    assert tuple(pred.shape) == cloud.shape
    arr = pred.numpy()
    assert np.all(np.isfinite(arr))


def test_residual_denoiser_with_staas_fusion():
    """STAAS fusion branch must produce a finite (B, N, 3) prediction."""
    import jittor as jt

    from denoise_baseline import ResidualDenoiser

    cloud = _make_noisy_cloud()
    model = ResidualDenoiser(
        k=8,
        feat_dim=32,
        hidden=32,
        use_staas=True,
        staas_fusion=True,
    )
    pred = model(jt.array(cloud))

    assert tuple(pred.shape) == cloud.shape
    assert np.all(np.isfinite(pred.numpy()))
