#!/usr/bin/env python3
"""GARA-D v0 synthetic smoke trainer/evaluator.

Purpose: verify whether a bounded directional adapter has learning signal before
any A6000 run or hidden-test submission work.

Synthetic setup:
  clean OBJ vertices -> sampled clean -> Gaussian noisy
  base prediction    -> deterministic local smoothing baseline
  GARA-D target      -> refine base toward clean with bounded residual

This does not use hidden test labels and does not claim leaderboard quality.
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

# Parse only early runtime flags before importing Jittor. Jittor may compile CUDA
# at import time, so --cpu must take effect before `import jittor`.
_early = argparse.ArgumentParser(add_help=False)
_early.add_argument("--cpu", action="store_true")
_early_args, _ = _early.parse_known_args()
if _early_args.cpu:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("DISABLE_MULTIPROCESSING", "1")
    os.environ["nvcc_path"] = ""

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jittor as jt
from jittor import nn


def load_obj_vertices(path: Path) -> np.ndarray:
    verts: list[list[float]] = []
    with path.open("r", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.strip().split()
            if len(parts) >= 4:
                try:
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                except ValueError:
                    continue
    pts = np.asarray(verts, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
        raise ValueError(f"bad obj vertices: {path}")
    return pts.astype(np.float32)


def normalize_pc(pc: np.ndarray) -> np.ndarray:
    pc = pc.astype(np.float32)
    center = pc.mean(axis=0, keepdims=True)
    pc = pc - center
    scale = np.sqrt((pc ** 2).sum(axis=1)).max()
    return pc / (scale + 1e-8)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)


def repo_path(path: str | Path | None) -> Path | None:
    if path is None or str(path) == "":
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def scalar(x: jt.Var) -> float:
    try:
        return float(x.item())
    except ModuleNotFoundError as e:
        if e.name == "cupy":
            return float("nan")
        raise


def chamfer_l2_np(a: np.ndarray, b: np.ndarray) -> float:
    ta = cKDTree(a)
    tb = cKDTree(b)
    da, _ = tb.query(a, k=1)
    db, _ = ta.query(b, k=1)
    return float((da ** 2).mean() + (db ** 2).mean())


def chamfer_l2_jt(a: jt.Var, b: jt.Var, max_points: int = 0) -> jt.Var:
    if max_points and max_points > 0:
        n = a.shape[1]
        if n > max_points:
            stride = max(1, n // max_points)
            a = a[:, ::stride, :]
            b = b[:, ::stride, :]
            if a.shape[1] > max_points:
                a = a[:, :max_points, :]
                b = b[:, :max_points, :]
    dist = ((a.unsqueeze(2) - b.unsqueeze(1)) ** 2).sum(dim=-1)
    return jt.min(dist, dim=2).mean() + jt.min(dist, dim=1).mean()


def metric_to_score(cd_pred: float, cd_base: float) -> float:
    if cd_base <= 1e-15:
        return 100.0 if cd_pred <= cd_base else 0.0
    return max(0.0, min(100.0, 100.0 * (1.0 - cd_pred / cd_base)))


def sample_points_rng(pc: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.choice(len(pc), n, replace=len(pc) < n)
    return pc[idx].astype(np.float32)


def list_obj_files(data_root: Path, list_file: Path, limit: int) -> list[tuple[str, Path]]:
    ids = [x.strip() for x in list_file.read_text().splitlines() if x.strip()]
    files: list[tuple[str, Path]] = []
    for sample in ids:
        rel = Path(sample)
        if rel.parts and rel.parts[0] == "shapenet":
            obj = data_root / rel / "models" / "model_normalized.obj"
        else:
            obj = data_root / "shapenet" / rel / "models" / "model_normalized.obj"
        if obj.exists():
            files.append((sample, obj))
        if limit > 0 and len(files) >= limit:
            break
    if not files:
        raise RuntimeError(f"no obj files found from {list_file} under {data_root}")
    return files


def smooth_base_prediction(noisy: np.ndarray, k: int = 12, alpha: float = 0.35) -> np.ndarray:
    """Cheap deterministic base prediction for synthetic smoke.

    It is intentionally imperfect but usually improves noisy CD a bit, giving the
    adapter a non-trivial base-refinement task without needing a heavy checkpoint.
    """
    # 这是训练前的“基线预测器”，不是正式模型；它只负责制造一个可学习的 base -> pred 任务。
    tree = cKDTree(noisy)
    _, idx = tree.query(noisy, k=k + 1)
    neigh = noisy[idx[:, 1:]]
    mean = neigh.mean(axis=1)
    base = (1.0 - alpha) * noisy + alpha * mean
    return base.astype(np.float32)


def synthetic_base_prediction(clean: np.ndarray, noisy: np.ndarray, rng: np.random.Generator, mode: str, k: int, alpha: float) -> np.ndarray:
    """Construct diagnostic base with controlled difficulty.

    The default `smooth` mode is fully noisy-only. The oracle-mix modes are
    synthetic-only diagnostics: they intentionally use clean to create base
    difficulty bands, so they must never be used as a hidden-test method.
    """
    # 这些 mode 都是 synthetic smoke 的诊断场景，不应被当作 hidden-test 方案。
    if mode in {"smooth", "paired-smooth"}:
        return smooth_base_prediction(noisy, k=k, alpha=alpha)

    smooth = smooth_base_prediction(noisy, k=k, alpha=alpha)
    residual_noise = noisy - clean
    if mode == "easy":
        # Clearly better than noisy, still has small residual error.
        base = clean + 0.35 * residual_noise + rng.normal(0.0, 0.0015, clean.shape).astype(np.float32)
    elif mode == "medium":
        # Better than noisy but leaves enough structured correction room.
        base = clean + 0.55 * residual_noise + 0.20 * (smooth - noisy)
    elif mode == "hard":
        # Close to noisy-only smoothing, with mild clean anchoring for stability.
        base = 0.75 * smooth + 0.25 * (clean + 0.65 * residual_noise)
    elif mode == "paired":
        # Oracle paired-offset sanity: known deterministic residual from clean.
        # This is deliberately easy and paired; if upper-bound MLP fails here,
        # the training/data loop is broken before any real GARA-D question.
        phase = rng.uniform(0.0, 2.0 * np.pi, size=(1, 3)).astype(np.float32)
        freq = rng.uniform(1.5, 3.5, size=(1, 3)).astype(np.float32)
        structured = np.sin(clean * freq * np.pi + phase).astype(np.float32)
        radial = clean / (np.sqrt((clean * clean).sum(axis=1, keepdims=True)) + 1e-6)
        residual = 0.0040 * structured + 0.0030 * radial
        residual += rng.normal(0.0, 0.0005, clean.shape).astype(np.float32)
        base = clean + residual
    elif mode == "paired-noise":
        # Paired noisy/local-geometry bridge. Keeps exact clean/base index
        # correspondence, but makes residual depend on actual noisy displacement,
        # smoothing bias, local scale, and radial direction. This is still
        # synthetic-only, but less oracle-clean than `paired`.
        radial = clean / (np.sqrt((clean * clean).sum(axis=1, keepdims=True)) + 1e-6)
        tree = cKDTree(clean)
        _, idx = tree.query(clean, k=min(k + 1, len(clean)))
        neigh = clean[idx[:, 1:]]
        local_scale = np.sqrt(np.maximum(((neigh - clean[:, None, :]) ** 2).sum(axis=-1), 0.0)).mean(axis=1, keepdims=True)
        local_scale = local_scale / (np.median(local_scale) + 1e-6)
        smooth_bias = smooth - noisy
        residual = 0.45 * residual_noise + 0.35 * smooth_bias
        residual += 0.0030 * np.tanh(local_scale) * radial
        residual += rng.normal(0.0, 0.0010, clean.shape).astype(np.float32)
        # Keep base better than noisy while leaving non-trivial paired correction.
        residual = np.clip(residual, -0.018, 0.018).astype(np.float32)
        base = clean + residual
    else:
        raise ValueError(f"unknown base mode: {mode}")
    return base.astype(np.float32)


def geometry_features_np(noisy: np.ndarray, base: np.ndarray, k: int = 16) -> np.ndarray:
    # 额外几何特征全部从 noisy/base 派生，用来模拟“只靠本地可得信息做门控”的现实约束。
    pts = noisy.astype(np.float32, copy=False)
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=k + 1)
    neigh = pts[idx[:, 1:]]
    rel = neigh - pts[:, None, :]
    dist = np.sqrt(np.maximum((rel * rel).sum(axis=-1), 0.0))
    scale = np.sort(dist, axis=1)[:, k // 2]
    centered = neigh - neigh.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / float(k)
    evals = np.linalg.eigvalsh(cov)
    l1 = np.maximum(evals[:, 2], 1e-12)
    l2 = np.maximum(evals[:, 1], 0.0)
    l3 = np.maximum(evals[:, 0], 0.0)
    linearity = np.clip((l1 - l2) / l1, 0.0, 1.0)
    scattering = np.clip(l3 / l1, 0.0, 1.0)
    edge_conf = np.clip(linearity * (1.0 - scattering), 0.0, 1.0)
    plane_res = np.sqrt(np.maximum(l3, 0.0))
    base_move = np.sqrt(np.maximum(((base - noisy) ** 2).sum(axis=-1), 0.0))
    feats = np.stack([scale, plane_res, edge_conf, scattering, base_move], axis=1).astype(np.float32)
    # Per-cloud robust normalization for scalar geometry features.
    med = np.median(feats, axis=0, keepdims=True)
    iqr = np.quantile(feats, 0.75, axis=0, keepdims=True) - np.quantile(feats, 0.25, axis=0, keepdims=True)
    feats = (feats - med) / (iqr + 1e-6)
    return np.clip(feats, -5.0, 5.0).astype(np.float32)


def make_sample(obj_path: Path, num_points: int, noise_min: float, noise_max: float, rng: np.random.Generator, base_k: int, base_alpha: float, geom_k: int, base_mode: str = "smooth") -> dict[str, np.ndarray]:
    # 每个样本都同时返回 noisy / clean / base / x，方便训练、评估和风险分析复用同一条样本链。
    clean_full = normalize_pc(load_obj_vertices(obj_path))
    clean = sample_points_rng(clean_full, num_points, rng)
    sigma = float(rng.uniform(noise_min, noise_max))
    noisy = clean + rng.normal(0.0, sigma, clean.shape).astype(np.float32)
    base = synthetic_base_prediction(clean, noisy, rng, base_mode, base_k, base_alpha)
    geom = geometry_features_np(noisy, base, k=geom_k)
    x = np.concatenate([noisy, base, noisy - base, geom], axis=1).astype(np.float32)
    return {"noisy": noisy.astype(np.float32), "clean": clean.astype(np.float32), "base": base, "x": x, "sigma": np.asarray([sigma], dtype=np.float32)}


class GARADv0(nn.Module):
    """Bounded directional residual adapter.

    Input per point: noisy xyz, base xyz, noisy-base xyz, geometry scalars.
    Output: pred = base + gate * sigmoid(distance) * max_step * normalized(direction).
    """

    def __init__(self, in_dim: int = 14, hidden: int = 96, max_step: float = 0.010) -> None:
        super().__init__()
        self.max_step = float(max_step)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
        )
        self.dir_head = nn.Linear(hidden // 2, 3)
        self.dist_head = nn.Linear(hidden // 2, 1)
        self.gate_head = nn.Linear(hidden // 2, 1)

    def execute(self, x: jt.Var, base: jt.Var, return_delta: bool = False):
        # 主输出是 residual delta，而不是直接重建 clean；这样更容易控制移动幅度。
        h = self.mlp(x)
        direction = self.dir_head(h)
        direction = direction / (((direction ** 2).sum(dim=-1, keepdims=True) + 1e-12) ** 0.5)
        distance = jt.sigmoid(self.dist_head(h)) * self.max_step
        gate = jt.sigmoid(self.gate_head(h))
        delta = gate * distance * direction
        pred = base + delta
        if return_delta:
            return pred, delta, gate, distance
        return pred


class ResidualMLPAdapter(nn.Module):
    """Unbounded-ish residual MLP upper-bound control.

    This is not the safe GARA-D design. It tests whether the synthetic setup has
    learnable correction signal at all. Delta is only softly bounded by tanh.
    """

    def __init__(self, in_dim: int = 14, hidden: int = 96, max_step: float = 0.010) -> None:
        super().__init__()
        self.max_step = float(max_step)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 3),
        )

    def execute(self, x: jt.Var, base: jt.Var, return_delta: bool = False):
        # 这个分支故意比 GARA-D 更“宽松”，用于判断 synthetic setup 是否真的有学习信号。
        delta = jt.tanh(self.mlp(x)) * self.max_step
        pred = base + delta
        gate = jt.ones((x.shape[0], x.shape[1], 1))
        distance = ((delta ** 2).sum(dim=-1, keepdims=True) + 1e-12) ** 0.5
        if return_delta:
            return pred, delta, gate, distance
        return pred


class ZeroAdapter(nn.Module):
    """Identity baseline control: pred == base."""

    def execute(self, x: jt.Var, base: jt.Var, return_delta: bool = False):
        # identity control：如果这个都不稳定，说明评估/打包链路本身有问题。
        delta = jt.zeros_like(base)
        gate = jt.zeros((base.shape[0], base.shape[1], 1))
        distance = jt.zeros((base.shape[0], base.shape[1], 1))
        if return_delta:
            return base, delta, gate, distance
        return base


def build_model(args: argparse.Namespace) -> nn.Module:
    if args.model == "garad":
        return GARADv0(in_dim=14, hidden=args.hidden, max_step=args.max_step)
    if args.model == "residual_mlp":
        return ResidualMLPAdapter(in_dim=14, hidden=args.hidden, max_step=args.max_step)
    if args.model == "zero":
        return ZeroAdapter()
    raise ValueError(f"unknown model: {args.model}")


def make_batch(samples: list[tuple[str, Path]], args: argparse.Namespace, rng: np.random.Generator) -> tuple[jt.Var, jt.Var, jt.Var, jt.Var]:
    xs, bases, cleans, noisys = [], [], [], []
    for _ in range(args.batch_size):
        _, obj = random.choice(samples)
        s = make_sample(obj, args.num_points, args.noise_min, args.noise_max, rng, args.base_k, args.base_alpha, args.geom_k, args.base_mode)
        xs.append(s["x"])
        bases.append(s["base"])
        cleans.append(s["clean"])
        noisys.append(s["noisy"])
    return jt.array(np.stack(xs)), jt.array(np.stack(bases)), jt.array(np.stack(cleans)), jt.array(np.stack(noisys))


def evaluate(model: GARADv0, samples: list[tuple[str, Path]], args: argparse.Namespace, out_csv: Path) -> dict[str, Any]:
    # 评估只用合成 clean/noisy/base，不碰 hidden test；目标是看 adapter 是否比 base 更好。
    rng = np.random.default_rng(args.seed + 9999)
    rows = []
    for i, (sample, obj) in enumerate(samples[: args.eval_limit], 1):
        s = make_sample(obj, args.num_points, args.noise_min, args.noise_max, rng, args.base_k, args.base_alpha, args.geom_k, args.base_mode)
        x = jt.array(s["x"][None, ...])
        base_j = jt.array(s["base"][None, ...])
        with jt.no_grad():
            pred_j, delta_j, gate_j, dist_j = model(x, base_j, return_delta=True)
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
    p = argparse.ArgumentParser(description="Train/evaluate GARA-D v0 on synthetic clean/noisy samples.")
    # 这套参数主要用于 smoke / ablation / upper-bound sanity，不应直接当正式训练配方。
    p.add_argument("--data-root", default="dataset_train")
    p.add_argument("--train-list", default="starter_code/datalist/train.txt")
    p.add_argument("--eval-list", default="starter_code/datalist/validate.txt")
    p.add_argument("--out-dir", default="analysis/garad_v0_synth_smoke_20260518")
    p.add_argument("--seed", type=int, default=20260518)
    p.add_argument("--train-limit", type=int, default=64)
    p.add_argument("--eval-limit", type=int, default=32)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-points", type=int, default=1024)
    p.add_argument("--noise-min", type=float, default=0.005)
    p.add_argument("--noise-max", type=float, default=0.02)
    p.add_argument("--base-k", type=int, default=12)
    p.add_argument("--base-alpha", type=float, default=0.35)
    p.add_argument("--base-mode", choices=["smooth", "paired-smooth", "easy", "medium", "hard", "paired", "paired-noise"], default="smooth")
    p.add_argument("--geom-k", type=int, default=16)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--model", choices=["garad", "residual_mlp", "zero"], default="garad")
    p.add_argument("--max-step", type=float, default=0.010)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--cd-num-points", type=int, default=512)
    p.add_argument("--lambda-offset", type=float, default=0.05)
    p.add_argument("--lambda-delta", type=float, default=0.0, help="Synthetic-only direct delta supervision weight: MSE(delta, clean-base).")
    p.add_argument("--lambda-cd", type=float, default=1.0, help="Weight for CD loss; set 0 for paired-offset sanity.")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    jt.flags.use_cuda = 0 if args.cpu else 1
    data_root = repo_path(args.data_root)
    train_list = repo_path(args.train_list)
    eval_list = repo_path(args.eval_list)
    out_dir = repo_path(args.out_dir)
    assert data_root and train_list and eval_list and out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_samples = list_obj_files(data_root, train_list, args.train_limit)
    eval_samples = list_obj_files(data_root, eval_list, args.eval_limit)
    model = build_model(args)
    trainable_params = list(model.parameters()) if hasattr(model, "parameters") else []
    opt = nn.Adam(trainable_params, lr=args.lr) if trainable_params else None
    rng = np.random.default_rng(args.seed)

    log_path = out_dir / "train_log.csv"
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "loss", "loss_cd", "loss_delta", "loss_offset", "delta_l2_mean", "gate_mean", "distance_mean", "elapsed_sec"])
        writer.writeheader()
        t0 = time.time()
        for step in range(1, args.steps + 1):
            # 训练目标同时覆盖 cd / delta / offset 三类信号，分别约束质量和移动幅度。
            x, base, clean, _noisy = make_batch(train_samples, args, rng)
            pred, delta, gate, dist = model(x, base, return_delta=True)
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

    ckpt = out_dir / f"{args.model}_v0.pkl"
    if trainable_params:
        model.save(str(ckpt))
    else:
        ckpt.write_text("zero adapter has no parameters\n")
    eval_csv = out_dir / "eval_synth.csv"
    eval_summary = evaluate(model, eval_samples, args, eval_csv)
    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "purpose": "GARA-D v0 synthetic smoke; verify learning signal only",
        "args": vars(args),
        "train_samples": len(train_samples),
        "eval_samples": len(eval_samples),
        "checkpoint": str(ckpt),
        "train_log": str(log_path),
        "eval_csv": str(eval_csv),
        "eval": eval_summary,
        "decision_hint": "Promising only if cd_pred < cd_base on average and pred_better_than_base_rate is clearly above 0.5.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    report = [
        "# GARA-D v0 synthetic smoke",
        "",
        "This is a synthetic local smoke test, not an official-score estimate.",
        "",
        f"- eval n: {eval_summary['n']}",
        f"- cd_noisy mean: {eval_summary['cd_noisy']:.8g}",
        f"- cd_base mean: {eval_summary['cd_base']:.8g}",
        f"- cd_pred mean: {eval_summary['cd_pred']:.8g}",
        f"- base_better_than_noisy_rate: {eval_summary['base_better_than_noisy_rate']:.3f}",
        f"- pred_better_than_base_rate: {eval_summary['pred_better_than_base_rate']:.3f}",
        f"- score_vs_base mean: {eval_summary['score_vs_base']:.3f}",
        f"- delta_l2_mean: {eval_summary['delta_l2_mean']:.8g}",
        f"- delta_l2_p95: {eval_summary['delta_l2_p95']:.8g}",
        f"- gate_mean: {eval_summary['gate_mean']:.8g}",
        "",
        "Decision rule: if this does not beat base in synthetic CD, fix model/features before any A6000 work.",
    ]
    (out_dir / "risk_report.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"out_dir": str(out_dir), "eval": eval_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
