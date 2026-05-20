"""Unit tests for the research model forward passes.

We exercise each model on a small CPU-friendly random cloud (B=2, N=64, C=3)
to catch interface regressions (wrong tensor rank, broken gather, missing
attributes). The tests do not assert numerical correctness — the goal is
shape stability and that ``execute`` runs end to end.

If Jittor cannot be imported in the current environment, the whole file is
skipped via ``pytestmark``.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import HAS_JITTOR, JITTOR_IMPORT_ERROR

pytestmark = pytest.mark.skipif(
    not HAS_JITTOR,
    reason=f"Jittor unavailable in this environment: {JITTOR_IMPORT_ERROR}",
)


# Tiny shapes — every test uses these so the JIT compile cache is reused.
BATCH = 2
NUM_POINTS = 64
FEAT_DIM = 32  # small but >= 16 so all MLP heads stay valid
K = 8


@pytest.fixture()
def small_points(random_pointcloud):
    """A small (B=2, N=64, 3) cloud sized to fit CPU smoke tests."""
    return random_pointcloud(batch=BATCH, num_points=NUM_POINTS, channels=3)


@pytest.fixture()
def small_features(random_pointcloud):
    """A small (B=2, N=64, FEAT_DIM) feature tensor for KNN-based layers."""
    return random_pointcloud(batch=BATCH, num_points=NUM_POINTS, channels=FEAT_DIM)


def test_pwsenel_forward_shape(small_points, small_features):
    """PWSENEL returns the same (B, N, C) shape as its feature input."""
    import jittor as jt

    from denoise_baseline import PWSENEL

    model = PWSENEL(channels=FEAT_DIM, k=K)
    out = model(jt.array(small_features), jt.array(small_points))

    assert tuple(out.shape) == (BATCH, NUM_POINTS, FEAT_DIM)


@pytest.mark.gpu
def test_pwsenel_v2_gate_returns_per_point_scalar(small_points, small_features):
    """PWSENELv2Gate returns a (B, N, 1) gate clamped to [0, 1].

    Marked gpu because PWSENELv2Gate's sigmoid/clamp chain triggers a Jittor
    CPU-backend segfault in some environments (no GPU needed numerically, but
    the compiled kernel path is unstable without CUDA).
    """
    import jittor as jt

    from denoise_baseline import PWSENELv2Gate

    model = PWSENELv2Gate(channels=FEAT_DIM, k=K)
    gate = model(jt.array(small_features), jt.array(small_points))

    assert tuple(gate.shape) == (BATCH, NUM_POINTS, 1)
    arr = gate.numpy()
    assert np.all(arr >= 0.0) and np.all(arr <= 1.0)


@pytest.mark.gpu
def test_pwsenel_v2_gate_return_stats_keys(small_points, small_features):
    """``return_stats=True`` yields the documented diagnostic dictionary.

    Marked gpu for the same CPU segfault reason as the scalar test above.
    """
    import jittor as jt

    from denoise_baseline import PWSENELv2Gate

    model = PWSENELv2Gate(channels=FEAT_DIM, k=K)
    gate, stats = model(
        jt.array(small_features), jt.array(small_points), return_stats=True
    )

    assert tuple(gate.shape) == (BATCH, NUM_POINTS, 1)
    assert set(stats.keys()) == {"noise_conf", "edge_conf", "move_gate"}
    for value in stats.values():
        assert tuple(value.shape) == (BATCH, NUM_POINTS, 1)


def test_staas_v0_forward_returns_pred_shape(small_points):
    """STAASv0 returns a (B, N, 3) refined-points tensor."""
    import jittor as jt

    from denoise_baseline import STAASv0

    model = STAASv0(k=K)
    pred = model(jt.array(small_points))

    assert tuple(pred.shape) == (BATCH, NUM_POINTS, 3)


def test_staas_v0_return_stats_has_geometry_fields(small_points):
    """STAASv0's stats dict carries all fields ResidualDenoiser consumes."""
    import jittor as jt

    from denoise_baseline import STAASv0

    model = STAASv0(k=K)
    pred, stats = model(jt.array(small_points), return_stats=True)

    assert tuple(pred.shape) == (BATCH, NUM_POINTS, 3)
    expected_keys = {
        "scale",
        "tau",
        "linearity",
        "planarity",
        "scattering",
        "edge_conf",
        "smooth_offset",
    }
    assert expected_keys.issubset(stats.keys())
    # smooth_offset is per-point 3D; the others are per-point scalars.
    assert tuple(stats["smooth_offset"].shape) == (BATCH, NUM_POINTS, 3)
    for key in ("scale", "tau", "linearity", "planarity", "scattering", "edge_conf"):
        assert tuple(stats[key].shape) == (BATCH, NUM_POINTS)
