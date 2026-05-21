"""Unit tests for low-risk training recipe switches.

These tests guard the knobs we use for GARA-D safe-plus style experiments:
loss-type selection and LR scheduling must be opt-in, config-driven, and keep
historical defaults unchanged.
"""

from __future__ import annotations

import pytest

from tests.conftest import HAS_BASELINE_DEPS, BASELINE_DEPS_ERROR

pytestmark = pytest.mark.skipif(
    not HAS_BASELINE_DEPS,
    reason=f"denoise_baseline deps unavailable: {BASELINE_DEPS_ERROR}",
)


def test_lr_factor_default_preserves_constant_lr():
    from denoise_baseline import lr_factor_for_step

    assert lr_factor_for_step(step=1, total_steps=1000) == pytest.approx(1.0)
    assert lr_factor_for_step(step=500, total_steps=1000) == pytest.approx(1.0)
    assert lr_factor_for_step(step=1000, total_steps=1000) == pytest.approx(1.0)


def test_lr_factor_warmup_cosine_schedule_shape():
    from denoise_baseline import lr_factor_for_step

    assert lr_factor_for_step(1, 100, scheduler="warmup_cosine", warmup_steps=10) == pytest.approx(0.1)
    assert lr_factor_for_step(10, 100, scheduler="warmup_cosine", warmup_steps=10) == pytest.approx(1.0)
    assert lr_factor_for_step(100, 100, scheduler="warmup_cosine", warmup_steps=10, eta_min_ratio=0.05) == pytest.approx(0.05)
    mid = lr_factor_for_step(55, 100, scheduler="warmup_cosine", warmup_steps=10, eta_min_ratio=0.05)
    assert 0.05 < mid < 1.0


def test_pointwise_offset_loss_mse_matches_historical_formula():
    import jittor as jt

    from denoise_baseline import pointwise_offset_loss

    pred = jt.array([[[0.0, 2.0, -2.0]]])
    clean = jt.array([[[1.0, 0.0, -1.0]]])
    got = float(pointwise_offset_loss(pred, clean, loss_type="mse").numpy().item())
    expected = float((((pred - clean) ** 2).mean()).numpy().item())

    assert got == pytest.approx(expected)


def test_pointwise_offset_loss_huber_caps_large_residuals():
    import jittor as jt

    from denoise_baseline import pointwise_offset_loss

    pred = jt.array([[[0.0, 0.02, -0.02]]])
    clean = jt.array([[[0.0, 0.0, 0.0]]])
    mse = float(pointwise_offset_loss(pred, clean, loss_type="mse").numpy().item())
    huber = float(pointwise_offset_loss(pred, clean, loss_type="huber", huber_delta=0.01).numpy().item())

    assert huber < mse


def test_parser_recipe_defaults_are_baseline_safe():
    from denoise_baseline import build_parser

    args = build_parser().parse_args([])

    assert args.loss_type == "mse"
    assert args.huber_delta == pytest.approx(0.01)
    assert args.lr_scheduler == "none"
    assert args.warmup_steps == 0
    assert args.eta_min_ratio == pytest.approx(0.05)


def test_apply_config_loads_recipe_switches(tmp_path):
    from denoise_baseline import apply_config, build_parser

    cfg = tmp_path / "recipe.yaml"
    cfg.write_text(
        "train:\n"
        "  loss_type: huber\n"
        "  huber_delta: 0.02\n"
        "  lr_scheduler: warmup_cosine\n"
        "  warmup_steps: 500\n"
        "  eta_min_ratio: 0.1\n"
    )

    parser = build_parser()
    args = parser.parse_args(["--config", str(cfg)])
    args._cli_overrides = {}
    args = apply_config(args)

    assert args.loss_type == "huber"
    assert args.huber_delta == pytest.approx(0.02)
    assert args.lr_scheduler == "warmup_cosine"
    assert args.warmup_steps == 500
    assert args.eta_min_ratio == pytest.approx(0.1)
