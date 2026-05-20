"""Unit tests for :mod:`denoise_utils`.

Focus is :func:`gather_neighbors`, which is the only public helper today and
is shared by every model in :mod:`denoise_baseline`. A regression here would
silently corrupt every KNN-based branch, so the tests cover shape, exact
values, and degenerate sizes.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import HAS_JITTOR, JITTOR_IMPORT_ERROR

pytestmark = pytest.mark.skipif(
    not HAS_JITTOR,
    reason=f"Jittor unavailable in this environment: {JITTOR_IMPORT_ERROR}",
)


def _numpy_gather(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Reference NumPy implementation mirroring ``gather_neighbors`` semantics."""
    B, N, C = x.shape
    K = idx.shape[-1]
    out = np.empty((B, N, K, C), dtype=x.dtype)
    for b in range(B):
        for n in range(N):
            for k in range(K):
                out[b, n, k] = x[b, idx[b, n, k]]
    return out


def test_gather_neighbors_output_shape(random_pointcloud, random_knn_idx):
    """Output shape must be (B, N, K, C) for (B, N, C) + (B, N, K) inputs."""
    import jittor as jt

    from denoise_utils import gather_neighbors

    x_np = random_pointcloud(batch=3, num_points=32, channels=5)
    idx_np = random_knn_idx(batch=3, num_points=32, k=8)

    out = gather_neighbors(jt.array(x_np), jt.array(idx_np))

    assert tuple(out.shape) == (3, 32, 8, 5)


def test_gather_neighbors_values_match_reference(random_pointcloud, random_knn_idx):
    """Exact values must equal the naive per-element NumPy gather."""
    import jittor as jt

    from denoise_utils import gather_neighbors

    x_np = random_pointcloud(batch=2, num_points=16, channels=4)
    idx_np = random_knn_idx(batch=2, num_points=16, k=6)

    out = gather_neighbors(jt.array(x_np), jt.array(idx_np))
    expected = _numpy_gather(x_np, idx_np)

    np.testing.assert_allclose(out.numpy(), expected, rtol=0, atol=0)


def test_gather_neighbors_known_indices():
    """Hand-built inputs verify the index math (no random fixtures)."""
    import jittor as jt

    from denoise_utils import gather_neighbors

    # Two batches, three points, two channels. Each point's features are
    # ``[batch_id * 10 + point_id, batch_id * 10 + point_id + 0.5]``, which
    # makes the expected gather trivially readable.
    x_np = np.array(
        [
            [[0.0, 0.5], [1.0, 1.5], [2.0, 2.5]],
            [[10.0, 10.5], [11.0, 11.5], [12.0, 12.5]],
        ],
        dtype=np.float32,
    )
    # For every point gather the same two neighbors (point 2 then point 0).
    idx_np = np.array(
        [
            [[2, 0], [2, 0], [2, 0]],
            [[2, 0], [2, 0], [2, 0]],
        ],
        dtype=np.int32,
    )

    out = gather_neighbors(jt.array(x_np), jt.array(idx_np)).numpy()

    # Batch 0: neighbor 2 -> [2.0, 2.5]; neighbor 0 -> [0.0, 0.5]
    np.testing.assert_allclose(out[0, 0, 0], [2.0, 2.5])
    np.testing.assert_allclose(out[0, 1, 1], [0.0, 0.5])
    # Batch 1 uses the same indices but its own row values.
    np.testing.assert_allclose(out[1, 0, 0], [12.0, 12.5])
    np.testing.assert_allclose(out[1, 2, 1], [10.0, 10.5])


@pytest.mark.parametrize("k", [1, 2, 16])
def test_gather_neighbors_varying_k(k, random_pointcloud, random_knn_idx):
    """Output shape must follow ``K`` for the degenerate K=1 case and beyond."""
    import jittor as jt

    from denoise_utils import gather_neighbors

    x_np = random_pointcloud(batch=2, num_points=20, channels=3)
    idx_np = random_knn_idx(batch=2, num_points=20, k=k)

    out = gather_neighbors(jt.array(x_np), jt.array(idx_np))

    assert tuple(out.shape) == (2, 20, k, 3)
    np.testing.assert_allclose(out.numpy(), _numpy_gather(x_np, idx_np))


def test_gather_neighbors_batch_one(random_pointcloud, random_knn_idx):
    """B=1 must not collapse the batch axis."""
    import jittor as jt

    from denoise_utils import gather_neighbors

    x_np = random_pointcloud(batch=1, num_points=10, channels=3)
    idx_np = random_knn_idx(batch=1, num_points=10, k=4)

    out = gather_neighbors(jt.array(x_np), jt.array(idx_np))

    assert tuple(out.shape) == (1, 10, 4, 3)
    np.testing.assert_allclose(out.numpy(), _numpy_gather(x_np, idx_np))


def test_gather_neighbors_self_index_returns_input(random_pointcloud):
    """When idx[b, n, 0] = n the first slot equals the input row."""
    import jittor as jt

    from denoise_utils import gather_neighbors

    x_np = random_pointcloud(batch=2, num_points=12, channels=3)
    self_idx = np.broadcast_to(
        np.arange(12, dtype=np.int32).reshape(1, 12, 1), (2, 12, 1)
    ).copy()

    out = gather_neighbors(jt.array(x_np), jt.array(self_idx)).numpy()

    np.testing.assert_allclose(out[:, :, 0, :], x_np)
