#!/usr/bin/env python3
"""Canonical VM-base Phase-1 audit for GARA-D.

This script tests whether GARA-D v0 can refine the official VM checkpoint output
on a clean/noisy validation proxy before any A6000 or fixed075-base work.

Hidden-test safety: uses only training/validation OBJ clean geometry to synthesize
noisy inputs and evaluate CD against clean. It does not use hidden labels.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

_early = argparse.ArgumentParser(add_help=False)
_early.add_argument("--cpu", action="store_true")
_early_args, _ = _early.parse_known_args()
if _early_args.cpu:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("DISABLE_MULTIPROCESSING", "1")
    os.environ["nvcc_path"] = ""

import numpy as np
import yaml
from scipy.spatial import cKDTree

# System Python has Jittor but not OmegaConf; VM code only needs
# OmegaConf.to_container when configs are not plain dicts. This audit passes
# plain dicts, so provide a tiny import stub instead of mutating env packages.
import types
if "omegaconf" not in sys.modules:
    omega_stub = types.ModuleType("omegaconf")
    class _OmegaConfStub:
        @staticmethod
        def to_container(x, resolve=True):
            return x
    omega_stub.OmegaConf = _OmegaConfStub
    sys.modules["omegaconf"] = omega_stub

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter_code"))

import jittor as jt
from jittor import nn

from scripts.train_garad_v0 import (  # reuse existing audited utilities/models
    GARADv0,
    ResidualMLPAdapter,
    ZeroAdapter,
    chamfer_l2_jt,
    chamfer_l2_np,
    geometry_features_np,
    list_obj_files,
    load_obj_vertices,
    metric_to_score,
    normalize_pc,
    repo_path,
    sample_points_rng,
    scalar,
    set_seed,
)
from starter_code.src.model.vm import VelocityModule, patch_based_denoise, patch_based_denoise_streaming


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        data = yaml.safe_load(f)
    return dict(data or {})


def build_vm_model(model_config: Path, transform_config: Path, ckpt: Path) -> VelocityModule:
    model_cfg = load_yaml(model_config)
    transform_cfg = load_yaml(transform_config)
    model = VelocityModule(model_cfg, transform_cfg)
    model.load(str(ckpt))
    model.eval()
    model.set_predict(True)
    return model


def vm_base_prediction(model: VelocityModule, noisy: np.ndarray, patch_size: int, seed_k: int, seed_k_alpha: int, streaming: bool) -> np.ndarray:
    x = jt.array(noisy.astype(np.float32))
    with jt.no_grad():
        if streaming:
            y = patch_based_denoise_streaming(model, x, patch_size=patch_size, seed_k=seed_k, seed_k_alpha=seed_k_alpha)
        else:
            y = patch_based_denoise(model, x, patch_size=patch_size, seed_k=seed_k, seed_k_alpha=seed_k_alpha)
    return np.asarray(y.numpy(), dtype=np.float32)


def make_vm_sample(obj_path: Path, vm_model: VelocityModule, args: argparse.Namespace, rng: np.random.Generator) -> dict[str, np.ndarray]:
    clean_full = normalize_pc(load_obj_vertices(obj_path))
    clean = sample_points_rng(clean_full, args.num_points, rng)
    sigma = float(rng.uniform(args.noise_min, args.noise_max))
    if args.noise_dist == "laplace":
        noise = rng.laplace(0.0, sigma, clean.shape).astype(np.float32)
    else:
        noise = rng.normal(0.0, sigma, clean.shape).astype(np.float32)
    noisy = clean + noise
    base = vm_base_prediction(
        vm_model,
        noisy,
        patch_size=args.vm_patch_size,
        seed_k=args.vm_seed_k,
        seed_k_alpha=args.vm_seed_k_alpha,
        streaming=args.vm_streaming,
    )
    geom = geometry_features_np(noisy, base, k=args.geom_k)
    x = np.concatenate([noisy, base, noisy - base, geom], axis=1).astype(np.float32)
    return {"noisy": noisy.astype(np.float32), "clean": clean.astype(np.float32), "base": base, "x": x, "sigma": np.asarray([sigma], dtype=np.float32)}


def build_adapter(args: argparse.Namespace) -> nn.Module:
    if args.model == "zero":
        return ZeroAdapter()
    if args.model == "residual_mlp":
        return ResidualMLPAdapter(in_dim=14, hidden=args.hidden, max_step=args.max_step)
    if args.model == "garad":
        return GARADv0(in_dim=14, hidden=args.hidden, max_step=args.max_step)
    raise ValueError(args.model)


def make_batch(samples: list[tuple[str, Path]], vm_model: VelocityModule, args: argparse.Namespace, rng: np.random.Generator):
    xs, bases, cleans, noisys = [], [], [], []
    for _ in range(args.batch_size):
        _, obj = random.choice(samples)
        s = make_vm_sample(obj, vm_model, args, rng)
        xs.append(s["x"])
        bases.append(s["base"])
        cleans.append(s["clean"])
        noisys.append(s["noisy"])
    return jt.array(np.stack(xs)), jt.array(np.stack(bases)), jt.array(np.stack(cleans)), jt.array(np.stack(noisys))


def evaluate(adapter: nn.Module, samples: list[tuple[str, Path]], vm_model: VelocityModule, args: argparse.Namespace, out_csv: Path) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed + 9999)
    rows: list[dict[str, Any]] = []
    for i, (sample, obj) in enumerate(samples[: args.eval_limit], 1):
        s = make_vm_sample(obj, vm_model, args, rng)
        x = jt.array(s["x"][None, ...])
        base_j = jt.array(s["base"][None, ...])
        with jt.no_grad():
            pred_j, delta_j, gate_j, dist_j = adapter(x, base_j, return_delta=True)
        pred = np.asarray(pred_j.numpy()[0], dtype=np.float32)
        delta = np.asarray(delta_j.numpy()[0], dtype=np.float32)
        gate = np.asarray(gate_j.numpy()[0], dtype=np.float32)
        dist = np.asarray(dist_j.numpy()[0], dtype=np.float32)
        cd_noisy = chamfer_l2_np(s["noisy"], s["clean"])
        cd_base = chamfer_l2_np(s["base"], s["clean"])
        cd_pred = chamfer_l2_np(pred, s["clean"])
        row = {
            "idx": i,
            "sample": sample,
            "sigma": float(s["sigma"][0]),
            "cd_noisy": cd_noisy,
            "cd_base": cd_base,
            "cd_pred": cd_pred,
            "base_better_than_noisy": int(cd_base < cd_noisy),
            "pred_better_than_base": int(cd_pred < cd_base),
            "pred_better_than_noisy": int(cd_pred < cd_noisy),
            "score_vs_base": metric_to_score(cd_pred, cd_base),
            "delta_l2_mean": float(np.sqrt((delta * delta).sum(axis=1)).mean()),
            "delta_l2_p95": float(np.quantile(np.sqrt((delta * delta).sum(axis=1)), 0.95)),
            "gate_mean": float(gate.mean()),
            "distance_mean": float(dist.mean()),
        }
        rows.append(row)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    means = {k: float(np.mean([r[k] for r in rows])) for k in ["cd_noisy", "cd_base", "cd_pred", "score_vs_base", "delta_l2_mean", "delta_l2_p95", "gate_mean", "distance_mean"]}
    means["pred_better_than_base_rate"] = float(np.mean([r["pred_better_than_base"] for r in rows]))
    means["base_better_than_noisy_rate"] = float(np.mean([r["base_better_than_noisy"] for r in rows]))
    means["n"] = len(rows)
    return means


def main() -> None:
    p = argparse.ArgumentParser(description="Audit GARA-D on canonical official VM-base proxy.")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--data-root", default="dataset_train")
    p.add_argument("--train-list", default="starter_code/datalist/train.txt")
    p.add_argument("--eval-list", default="starter_code/datalist/validate.txt")
    p.add_argument("--out-dir", default="analysis/garad_v01_vm_base_phase1_audit_20260519")
    p.add_argument("--seed", type=int, default=20260519)
    p.add_argument("--train-limit", type=int, default=16)
    p.add_argument("--eval-limit", type=int, default=24)
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-points", type=int, default=512)
    p.add_argument("--noise-min", type=float, default=0.005)
    p.add_argument("--noise-max", type=float, default=0.020)
    p.add_argument("--noise-dist", choices=["laplace", "gaussian"], default="laplace")
    p.add_argument("--geom-k", type=int, default=16)
    p.add_argument("--model", choices=["zero", "residual_mlp", "garad"], default="garad")
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--max-step", type=float, default=0.010)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--cd-num-points", type=int, default=512)
    p.add_argument("--lambda-offset", type=float, default=0.05)
    p.add_argument("--lambda-delta", type=float, default=0.0)
    p.add_argument("--lambda-cd", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--vm-ckpt", default="starter_code/experiments/vm_quick/checkpoint_0.pkl")
    p.add_argument("--vm-model-config", default="starter_code/configs/model/vm.yaml")
    p.add_argument("--vm-transform-config", default="starter_code/configs/transform/vm.yaml")
    p.add_argument("--vm-patch-size", type=int, default=256)
    p.add_argument("--vm-seed-k", type=int, default=6)
    p.add_argument("--vm-seed-k-alpha", type=int, default=1)
    p.add_argument("--vm-streaming", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    jt.flags.use_cuda = 0 if args.cpu else 1

    data_root = repo_path(args.data_root)
    train_list = repo_path(args.train_list)
    eval_list = repo_path(args.eval_list)
    out_dir = repo_path(args.out_dir)
    vm_ckpt = repo_path(args.vm_ckpt)
    vm_model_config = repo_path(args.vm_model_config)
    vm_transform_config = repo_path(args.vm_transform_config)
    assert data_root and train_list and eval_list and out_dir and vm_ckpt and vm_model_config and vm_transform_config
    out_dir.mkdir(parents=True, exist_ok=True)

    train_samples = list_obj_files(data_root, train_list, args.train_limit)
    eval_samples = list_obj_files(data_root, eval_list, args.eval_limit)
    vm_model = build_vm_model(vm_model_config, vm_transform_config, vm_ckpt)
    adapter = build_adapter(args)
    trainable_params = list(adapter.parameters()) if hasattr(adapter, "parameters") else []
    opt = nn.Adam(trainable_params, lr=args.lr) if trainable_params else None
    rng = np.random.default_rng(args.seed)

    log_path = out_dir / "train_log.csv"
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "loss", "loss_cd", "loss_delta", "loss_offset", "delta_l2_mean", "gate_mean", "distance_mean", "elapsed_sec"])
        writer.writeheader()
        t0 = time.time()
        for step in range(1, args.steps + 1):
            x, base, clean, _noisy = make_batch(train_samples, vm_model, args, rng)
            pred, delta, gate, dist = adapter(x, base, return_delta=True)
            loss_cd = chamfer_l2_jt(pred, clean, args.cd_num_points)
            delta_l2 = ((delta ** 2).sum(dim=-1) + 1e-12) ** 0.5
            loss_offset = (delta_l2 ** 2).mean()
            target_delta = clean - base
            loss_delta = ((delta - target_delta) ** 2).mean()
            loss = args.lambda_cd * loss_cd + args.lambda_offset * loss_offset + args.lambda_delta * loss_delta
            if opt is not None:
                opt.step(loss)
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                row = {
                    "step": step,
                    "loss": scalar(loss),
                    "loss_cd": scalar(loss_cd),
                    "loss_delta": scalar(loss_delta),
                    "loss_offset": scalar(loss_offset),
                    "delta_l2_mean": scalar(delta_l2.mean()),
                    "gate_mean": scalar(gate.mean()),
                    "distance_mean": scalar(dist.mean()),
                    "elapsed_sec": round(time.time() - t0, 3),
                }
                writer.writerow(row)
                f.flush()
                print(row)

    ckpt = out_dir / f"{args.model}_vm_base_phase1.pkl"
    if trainable_params:
        adapter.save(str(ckpt))
    else:
        ckpt.write_text("zero adapter has no parameters\n")
    eval_csv = out_dir / "eval_vm_base.csv"
    eval_summary = evaluate(adapter, eval_samples, vm_model, args, eval_csv)
    pass_gate = bool(eval_summary["cd_pred"] < eval_summary["cd_base"] and eval_summary["pred_better_than_base_rate"] >= 0.60)
    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "purpose": "Canonical VM-base Phase-1 audit for GARA-D/residual adapters",
        "args": vars(args),
        "train_samples": len(train_samples),
        "eval_samples": len(eval_samples),
        "checkpoint": str(ckpt),
        "train_log": str(log_path),
        "eval_csv": str(eval_csv),
        "eval": eval_summary,
        "pass_gate": pass_gate,
        "decision_rule": "Pass only if cd_pred < cd_base and pred_better_than_base_rate >= 0.60; compare GARA-D against residual_mlp before A6000/fixed075.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# VM-base Phase-1 audit",
        "",
        f"model: `{args.model}`",
        f"seed: `{args.seed}`",
        f"vm_ckpt: `{args.vm_ckpt}`",
        "",
        f"- cd_noisy: {eval_summary['cd_noisy']:.8g}",
        f"- cd_base: {eval_summary['cd_base']:.8g}",
        f"- cd_pred: {eval_summary['cd_pred']:.8g}",
        f"- base_better_than_noisy_rate: {eval_summary['base_better_than_noisy_rate']:.3f}",
        f"- pred_better_than_base_rate: {eval_summary['pred_better_than_base_rate']:.3f}",
        f"- delta_l2_mean: {eval_summary['delta_l2_mean']:.8g}",
        f"- delta_l2_p95: {eval_summary['delta_l2_p95']:.8g}",
        f"- pass_gate: {int(pass_gate)}",
    ]
    (out_dir / "diagnosis.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"out_dir": str(out_dir), "eval": eval_summary, "pass_gate": pass_gate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
