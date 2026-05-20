#!/usr/bin/env python3
"""
Public utilities shared across the Jittor point-cloud denoising baseline.

This module currently exposes a single helper:
- :func:`gather_neighbors` — gather per-point neighbor features along a KNN
  index tensor. The implementation was previously duplicated in several model
  classes (``PWSENEL``, ``PWSENELv2Gate``, ``STAASv0``, ``ResidualDenoiser``)
  inside ``denoise_baseline.py``; the four copies were byte-identical, so they
  are unified here without changing any numerical behavior.

Keeping this helper standalone makes it easy to reuse from other research
scripts (e.g. probes under ``analysis/`` or ``scripts/``) and to test in
isolation without importing the whole training script.
"""

from __future__ import annotations

import jittor as jt


def gather_neighbors(x: jt.Var, idx: jt.Var) -> jt.Var:
    """Gather neighbor features by index along the second axis.

    Parameters
    ----------
    x : jt.Var
        Per-point features with shape ``(B, N, C)``.
    idx : jt.Var
        KNN indices with shape ``(B, N, K)``. Each entry ``idx[b, n, k]`` must
        lie in ``[0, N)`` and refers to a row of ``x[b]``.

    Returns
    -------
    jt.Var
        Tensor of shape ``(B, N, K, C)`` where
        ``out[b, n, k] = x[b, idx[b, n, k]]``.
    """
    # x: (B,N,C), idx: (B,N,K) -> (B,N,K,C)
    B, N, C = x.shape
    base = (jt.arange(B) * N).reshape(B, 1, 1)
    flat_idx = (idx + base).reshape(-1)
    flat = x.reshape(B * N, C)
    return flat[flat_idx].reshape(B, N, idx.shape[-1], C)
