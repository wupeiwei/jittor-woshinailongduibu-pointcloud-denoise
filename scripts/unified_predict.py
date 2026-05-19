#!/usr/bin/env python3
"""Unified inference entry point for formal A/B denoising candidates.

The old project had multiple ad-hoc entry points (plain predict and router
predict).  This script is the Phase-1 bridge toward `UnifiedDenoisePipeline`:
one command owns noisy stats, optional hard router, patch/chunk inference,
submission writing, route logging, and optional Candidate Registry append.

It does not train models and does not evaluate by default.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ResidualDenoiser = Any


def load_denoise_runtime():
    """Import Jittor-dependent runtime only after CLI parsing.

    This keeps `scripts/unified_predict.py --help` and static tooling from
    initializing Jittor or touching CUDA.
    """
    # Jittor 的 import 可能触发编译和 CUDA 探测，所以必须延迟到 CLI 解析之后。
    # 这样 `--help`、静态扫描、候选登记等纯 Python 操作不会被 CUDA 环境拖挂。
    import jittor as jt  # noqa: PLC0415
    from denoise_baseline import ResidualDenoiser, deep_update, predict_points_in_chunks  # noqa: PLC0415

    return jt, ResidualDenoiser, deep_update, predict_points_in_chunks


def deep_update_dict(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    # profile 覆盖 config 时只覆盖声明过的叶子字段，未声明字段沿用主配置。
    for key, value in (patch or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update_dict(base[key], value)
        else:
            base[key] = value
    return base


def repo_path(path: str | Path | None) -> Path | None:
    if path is None or str(path) == "":
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_yaml(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def load_config(config: Path, profiles: list[Path]) -> dict[str, Any]:
    cfg = read_yaml(config)
    for profile in profiles:
        cfg = deep_update_dict(cfg, read_yaml(profile))
    return cfg


def build_model(runtime: dict[str, Any], cfg: dict[str, Any], ckpt: Path):
    model_cls = runtime["ResidualDenoiser"]
    model_cfg = cfg.get("model", {})
    # 这里显式列出所有模型开关，避免不同候选配置隐式共享错误默认值。
    # 新增研究开关时也应在这里补齐，否则 unified_predict 可能复现不了训练配置。
    model = model_cls(
        k=int(model_cfg.get("k", 16)),
        feat_dim=int(model_cfg.get("feat_dim", 256)),
        hidden=int(model_cfg.get("hidden", 256)),
        use_pwsenel=bool(model_cfg.get("pwsenel", False)),
        use_staas=bool(model_cfg.get("staas", False)),
        staas_strength=float(model_cfg.get("staas_strength", 1.0)),
        staas_tau0=float(model_cfg.get("staas_tau0", 0.02)),
        staas_tau_min=float(model_cfg.get("staas_tau_min", 0.005)),
        staas_tau_max=float(model_cfg.get("staas_tau_max", 0.08)),
        use_move_gate=bool(model_cfg.get("move_gate", False)),
        use_pwsenel_v2=bool(model_cfg.get("pwsenel_v2", False)),
        pwsenel_v2_edge_lock=float(model_cfg.get("pwsenel_v2_edge_lock", 0.7)),
        pwsenel_v2_gate_scale=float(model_cfg.get("pwsenel_v2_gate_scale", 0.5)),
        residual_clip=float(model_cfg.get("residual_clip", 0.0)),
        adaptive_clip=bool(model_cfg.get("adaptive_clip", False)),
        adaptive_clip_min=float(model_cfg.get("adaptive_clip_min", 0.006)),
        adaptive_clip_max=float(model_cfg.get("adaptive_clip_max", 0.020)),
        adaptive_clip_ref_low=float(model_cfg.get("adaptive_clip_ref_low", 0.022)),
        adaptive_clip_ref_mid=float(model_cfg.get("adaptive_clip_ref_mid", 0.030)),
        adaptive_clip_ref_high=float(model_cfg.get("adaptive_clip_ref_high", 0.040)),
        adaptive_clip_mid=float(model_cfg.get("adaptive_clip_mid", 0.010)),
        noise_aware_move_gate=bool(model_cfg.get("noise_aware_move_gate", False)),
        noise_aware_gate_min=float(model_cfg.get("noise_aware_gate_min", 0.45)),
        noise_aware_gate_ref_low=float(model_cfg.get("noise_aware_gate_ref_low", 0.022)),
        noise_aware_gate_ref_high=float(model_cfg.get("noise_aware_gate_ref_high", 0.036)),
        hybrid_safe_strong=bool(model_cfg.get("hybrid_safe_strong", False)),
        hybrid_router_scale=float(model_cfg.get("hybrid_router_scale", 1.0)),
    )
    model.load(str(ckpt))
    model.eval()
    return model


def plane_res_p75(points: np.ndarray, k: int = 16, max_points: int = 8192, seed: int = 1234) -> float:
    # 隐藏测试没有 GT，只能用 noisy 自身估计难度；平面残差 p75 是路由用的保守粗指标。
    # 为了控制大点云成本，最多抽样 max_points，且 seed 按样本变化以便可复现。
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) > max_points:
        rng = np.random.default_rng(seed)
        pts = pts[rng.choice(len(pts), max_points, replace=False)]
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=k + 1)
    idx = idx[:, 1:]
    neigh = pts[idx]
    centered = neigh - neigh.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / max(k, 1)
    evals = np.linalg.eigvalsh(cov)
    residual = np.sqrt(np.maximum(evals[:, 0], 0.0))
    return float(np.quantile(residual, 0.75))


def make_zip(out_dir: Path, zip_path: Path) -> int:
    # 官方提交 zip 只接受 shapenet/<category>/<model>/denoised.npy 这种相对路径。
    # 这里从 out_dir 重新打包，避免把本机绝对路径或中间目录写进提交包。
    files = sorted(out_dir.glob("shapenet/*/*/denoised.npy"))
    if not files:
        raise RuntimeError(f"no denoised.npy files under {out_dir}")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, f.relative_to(out_dir))
    print(f"wrote {rel(zip_path)} files={len(files)}")
    return len(files)


class LIRDenoiserV0:
    """Phase-2 minimal local iterative refinement wrapper.

    It reuses one denoiser checkpoint and applies it repeatedly with a convex
    interpolation update:

        x_{t+1} = (1 - alpha) * x_t + alpha * f(x_t)

    The goal is not to invent a new backbone yet, only to verify whether a
    tiny fixed-point refinement loop improves low-noise stability without
    changing the model weights.
    """

    def __init__(self, model, patch_size: int, steps: int = 2, alpha: float = 0.5, predict_fn=None) -> None:
        self.model = model
        self.patch_size = int(patch_size)
        self.steps = max(1, int(steps))
        self.alpha = float(alpha)
        self.predict_fn = predict_fn

    def _predict_once(self, noisy_np: np.ndarray) -> np.ndarray:
        return self.predict_fn(self.model, noisy_np, self.patch_size)

    def predict(self, noisy_np: np.ndarray, sample_index: int) -> tuple[np.ndarray, dict[str, Any]]:
        # LIR 是“同一 checkpoint 多次固定点迭代”的实验路径；
        # 每一步都用凸组合，避免直接把高方差预测整量替换当前点云。
        current = np.asarray(noisy_np, dtype=np.float32)
        deltas: list[float] = []
        for step in range(self.steps):
            pred = self._predict_once(current).astype(np.float32)
            delta = float(np.mean(np.linalg.norm(pred - current, axis=1))) if len(current) else 0.0
            deltas.append(delta)
            current = ((1.0 - self.alpha) * current + self.alpha * pred).astype(np.float32)
        return current.astype(np.float32), {
            "mode": "lir",
            "route": "lir",
            "hard_route": "lir",
            "gate": self.alpha,
            "lir_steps": self.steps,
            "lir_alpha": self.alpha,
            "mean_step_delta_l2": float(np.mean(deltas)) if deltas else 0.0,
            "final_step_delta_l2": float(deltas[-1]) if deltas else 0.0,
        }


class GatedLIRDenoiserV0:
    """Noisy-only gate around LIR-Denoiser v0.

    Low estimated-noise samples use the safe one-shot denoiser; higher-noise
    samples use the lightweight iterative refinement wrapper. This keeps the
    A/B-compatible single inference entry while avoiding the low-noise failures
    seen in synthetic CD evaluation.
    """

    def __init__(
        self,
        model,
        patch_size: int,
        threshold: float,
        estimator_k: int,
        estimator_max_points: int,
        seed: int,
        steps: int = 2,
        alpha: float = 0.75,
        predict_fn=None,
    ) -> None:
        self.model = model
        self.patch_size = int(patch_size)
        self.threshold = float(threshold)
        self.estimator_k = int(estimator_k)
        self.estimator_max_points = int(estimator_max_points)
        self.seed = int(seed)
        self.steps = max(1, int(steps))
        self.alpha = float(alpha)
        self.predict_fn = predict_fn
        self.lir = LIRDenoiserV0(
            model=model,
            patch_size=patch_size,
            steps=steps,
            alpha=alpha,
            predict_fn=predict_fn,
        )

    def _predict_safe(self, noisy_np: np.ndarray) -> np.ndarray:
        return self.predict_fn(self.model, noisy_np, self.patch_size).astype(np.float32)

    def predict(self, noisy_np: np.ndarray, sample_index: int) -> tuple[np.ndarray, dict[str, Any]]:
        # gated-lir 的关键边界：只用 noisy 统计量决定是否进入 LIR，
        # 不读取 GT，也不依赖 leaderboard 反馈，因此可以用于隐藏测试风险排查。
        stat = plane_res_p75(
            noisy_np,
            k=self.estimator_k,
            max_points=self.estimator_max_points,
            seed=self.seed + sample_index,
        )
        if stat < self.threshold:
            pred = self._predict_safe(noisy_np)
            return pred, {
                "mode": "gated-lir",
                "plane_res_p75": stat,
                "route": "safe",
                "hard_route": "safe",
                "gate": 0.0,
                "lir_steps": 0,
                "lir_alpha": "",
                "mean_step_delta_l2": "",
                "final_step_delta_l2": "",
            }
        pred, info = self.lir.predict(noisy_np, sample_index)
        info.update({
            "mode": "gated-lir",
            "plane_res_p75": stat,
            "route": "lir",
            "hard_route": "lir",
            "gate": 1.0,
        })
        return pred, info


class NoisyConditionedDenoiser:
    """Phase-1 noisy-conditioned router wrapper.

    This class intentionally exposes the formal interface expected by the plan.
    It supports hard routing, soft blending, and forced wrong-route stress modes
    behind the same CLI/submission writer.
    """

    def __init__(
        self,
        safe_model: ResidualDenoiser,
        strong_model,
        threshold: float,
        patch_size: int,
        estimator_k: int,
        estimator_max_points: int,
        seed: int,
        predict_fn,
        router_mode: str = "hard",
        soft_width: float = 0.001,
    ) -> None:
        self.safe_model = safe_model
        self.strong_model = strong_model or safe_model
        self.threshold = threshold
        self.patch_size = patch_size
        self.estimator_k = estimator_k
        self.estimator_max_points = estimator_max_points
        self.seed = seed
        self.predict_fn = predict_fn
        self.router_mode = router_mode
        self.soft_width = max(float(soft_width), 1e-8)

    def _predict_model(self, model, noisy_np: np.ndarray) -> np.ndarray:
        return self.predict_fn(model, noisy_np, self.patch_size)

    def predict(self, noisy_np: np.ndarray, sample_index: int) -> tuple[np.ndarray, dict[str, Any]]:
        # router 模式保留 safe/strong 两个分支的统一封装；
        # force-* 只用于错路压力测试，hard/soft 才是正常路由实验。
        stat = plane_res_p75(
            noisy_np,
            k=self.estimator_k,
            max_points=self.estimator_max_points,
            seed=self.seed + sample_index,
        )
        hard_route = "safe" if stat < self.threshold else "strong"

        if self.router_mode == "force-safe":
            pred = self._predict_model(self.safe_model, noisy_np)
            route = "safe"
            gate = 0.0
        elif self.router_mode == "force-strong":
            pred = self._predict_model(self.strong_model, noisy_np)
            route = "strong"
            gate = 1.0
        elif self.router_mode == "soft":
            gate = float(1.0 / (1.0 + np.exp(-(stat - self.threshold) / self.soft_width)))
            safe_pred = self._predict_model(self.safe_model, noisy_np).astype(np.float32)
            strong_pred = self._predict_model(self.strong_model, noisy_np).astype(np.float32)
            if safe_pred.shape != strong_pred.shape:
                raise RuntimeError(f"soft router shape mismatch: safe={safe_pred.shape}, strong={strong_pred.shape}")
            pred = (1.0 - gate) * safe_pred + gate * strong_pred
            route = "soft"
        elif self.router_mode == "hard":
            route = hard_route
            gate = 0.0 if route == "safe" else 1.0
            model = self.safe_model if route == "safe" else self.strong_model
            pred = self._predict_model(model, noisy_np)
        else:
            raise ValueError(f"unknown router_mode: {self.router_mode}")

        return pred, {
            "plane_res_p75": stat,
            "route": route,
            "hard_route": hard_route,
            "gate": gate,
        }


def run_registry(args: argparse.Namespace, zip_path: Path, candidate_mode: str) -> None:
    # 推理完成后可选地追加 Candidate Registry。这里不重新评估分数，
    # 只把当前 artifact、配置、路由参数和结论登记成可追溯记录。
    branch = "UnifiedDenoisePipeline router"
    stage = "Phase 1"
    notes = ""
    if candidate_mode == "lir":
        branch = "UnifiedDenoisePipeline LIR-Denoiser v0"
        stage = "Phase 2"
        notes = f"lir_steps={args.lir_steps}; lir_alpha={args.lir_alpha}"
    elif candidate_mode == "gated-lir":
        branch = "UnifiedDenoisePipeline gated LIR-Denoiser v0"
        stage = "Phase 2"
        notes = f"threshold={args.threshold}; lir_steps={args.lir_steps}; lir_alpha={args.lir_alpha}"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "candidate_registry.py"),
        "--name",
        args.name,
        "--stage",
        stage,
        "--status",
        "candidate",
        "--config",
        args.safe_config,
        "--ckpt",
        args.safe_ckpt,
        "--zip",
        str(zip_path),
        "--branch",
        branch,
        "--router-threshold",
        str(args.threshold),
        "--patch-size",
        str(args.patch_size),
        "--chunk-size",
        str(args.patch_size),
        "--overlap",
        "0",
        "--stitching-strategy",
        "contiguous_chunks",
        "--submission-check",
        "pending",
        "--conclusion",
        args.conclusion,
    ]
    if candidate_mode in {"lir", "gated-lir"}:
        cmd.extend(["--soft-gate-temperature", "", "--notes", notes])
    for profile in args.profile:
        cmd.extend(["--profile", profile])
    if args.strong_ckpt:
        extra_notes = f"strong_config={args.strong_config}; strong_ckpt={args.strong_ckpt}"
        if notes:
            notes = f"{notes}; {extra_notes}"
        else:
            notes = extra_notes
        cmd.extend(["--notes", notes])
    subprocess.check_call(cmd, cwd=ROOT)


def main() -> None:
    p = argparse.ArgumentParser(description="UnifiedDenoisePipeline inference/submission entry.")
    # 默认值保留历史 router 候选，当前是否推荐提交以 registry/docs 为准；
    # 正式复现应显式传入 config/ckpt/name，避免误用旧默认候选。
    p.add_argument("--name", default="router_t0165")
    p.add_argument("--safe-config", default="configs/denoise_pwsenel_v2_adaptive_clip_piecewise.yaml")
    p.add_argument("--strong-config", default="configs/denoise_noise_aware_move_gate.yaml")
    p.add_argument("--profile", action="append", default=[])
    p.add_argument("--safe-ckpt", default="experiments/denoise_pwsenel_v2_adaptive_clip_piecewise/pwsenel_v2_adaptive_clip_piecewise.pkl")
    p.add_argument("--strong-ckpt", default="experiments/denoise_noise_aware_move_gate/noise_aware_move_gate.pkl")
    p.add_argument("--test-root", default="dataset_test_noisy")
    p.add_argument("--out-dir", default="results/denoise_router_t0165")
    p.add_argument("--zip", default="result_denoise_router_t0165.zip")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--patch-size", type=int, default=8192)
    p.add_argument("--estimator-k", type=int, default=16)
    p.add_argument("--estimator-max-points", type=int, default=8192)
    p.add_argument("--threshold", type=float, default=0.0165, help="safe branch if plane_res_p75 < threshold")
    p.add_argument(
        "--router-mode",
        choices=["hard", "soft", "force-safe", "force-strong"],
        default="hard",
        help="Routing mode: hard threshold, soft blend, or forced branch for wrong-route stress.",
    )
    p.add_argument("--soft-width", type=float, default=0.001, help="Soft router sigmoid width around threshold.")
    p.add_argument("--mode", choices=["router", "lir", "gated-lir"], default="router")
    p.add_argument("--lir-steps", type=int, default=2)
    p.add_argument("--lir-alpha", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-zip", action="store_true")
    p.add_argument("--append-registry", action="store_true")
    p.add_argument("--conclusion", default="Phase 2 LIR-Denoiser v0 smoke candidate; official score pending.")
    args = p.parse_args()

    # 到这里才加载 Jittor 运行时，确保纯参数解析不会触发 GPU/CUDA 编译。
    jt, model_cls, _deep_update, predict_fn = load_denoise_runtime()
    runtime = {
        "jt": jt,
        "ResidualDenoiser": model_cls,
        "predict_points_in_chunks": predict_fn,
    }
    jt.flags.use_cuda = 0 if args.cpu else 1
    profiles = [repo_path(x) for x in args.profile]
    profiles = [x for x in profiles if x is not None]
    safe_config = repo_path(args.safe_config)
    strong_config = repo_path(args.strong_config)
    safe_ckpt = repo_path(args.safe_ckpt)
    strong_ckpt = repo_path(args.strong_ckpt) if args.strong_ckpt else None
    test_root = repo_path(args.test_root)
    out_dir = repo_path(args.out_dir)
    zip_path = repo_path(args.zip)
    assert safe_config and strong_config and safe_ckpt and test_root and out_dir and zip_path

    for required in [safe_config, strong_config, safe_ckpt, test_root]:
        if not required.exists():
            raise FileNotFoundError(str(required))
    if strong_ckpt is not None and not strong_ckpt.exists():
        raise FileNotFoundError(str(strong_ckpt))

    print(f"loading safe:   {rel(safe_ckpt)}")
    safe_model = build_model(runtime, load_config(safe_config, profiles), safe_ckpt)

    strong_model = None
    if strong_ckpt is not None:
        print(f"loading strong: {rel(strong_ckpt)}")
        strong_model = build_model(runtime, load_config(strong_config, profiles), strong_ckpt)


    pipeline: Any
    if args.mode == "lir":
        pipeline = LIRDenoiserV0(
            model=safe_model,
            patch_size=args.patch_size,
            steps=args.lir_steps,
            alpha=args.lir_alpha,
            predict_fn=predict_fn,
        )
    elif args.mode == "gated-lir":
        pipeline = GatedLIRDenoiserV0(
            model=safe_model,
            patch_size=args.patch_size,
            threshold=args.threshold,
            estimator_k=args.estimator_k,
            estimator_max_points=args.estimator_max_points,
            seed=args.seed,
            steps=args.lir_steps,
            alpha=args.lir_alpha,
            predict_fn=predict_fn,
        )
    else:
        pipeline = NoisyConditionedDenoiser(
            safe_model=safe_model,
            strong_model=strong_model,
            threshold=args.threshold,
            patch_size=args.patch_size,
            estimator_k=args.estimator_k,
            estimator_max_points=args.estimator_max_points,
            seed=args.seed,
            predict_fn=predict_fn,
            router_mode=args.router_mode,
            soft_width=args.soft_width,
        )

    files = sorted(test_root.glob("shapenet/*/*/noisy.npy"))
    if args.limit > 0:
        files = files[:args.limit]
    if not files:
        raise RuntimeError(f"no noisy.npy files under {test_root}")

    out_dir.mkdir(parents=True, exist_ok=True)
    route_csv = out_dir / "routes.csv"
    fields = [
        "idx",
        "input",
        "output",
        "num_points",
        "plane_res_p75",
        "route",
        "hard_route",
        "gate",
        "lir_steps",
        "lir_alpha",
        "mean_step_delta_l2",
        "final_step_delta_l2",
        "patch_size",
        "elapsed_sec",
    ]
    counts = {"safe": 0, "strong": 0}
    t_all = time.time()
    with route_csv.open("w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fields)
        writer.writeheader()
        for i, f in enumerate(files, 1):
            # 每个样本独立写 denoised.npy 和 routes.csv 行；
            # 如果长任务中断，已经完成的样本和路由日志仍可用于排查。
            t0 = time.time()
            noisy_np = np.load(f).astype(np.float32)
            pred, info = pipeline.predict(noisy_np, i)
            rel_in = f.relative_to(test_root)
            out = out_dir / rel_in.parent / "denoised.npy"
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out, pred.astype(np.float32))
            route = str(info["route"])
            counts[route] = counts.get(route, 0) + 1
            elapsed = time.time() - t0
            writer.writerow({
                "idx": i,
                "input": str(rel_in),
                "output": str(out.relative_to(out_dir)),
                "num_points": len(noisy_np),
                "plane_res_p75": info.get("plane_res_p75", ""),
                "route": route,
                "hard_route": info.get("hard_route", ""),
                "gate": info.get("gate", ""),
                "lir_steps": info.get("lir_steps", ""),
                "lir_alpha": info.get("lir_alpha", ""),
                "mean_step_delta_l2": info.get("mean_step_delta_l2", ""),
                "final_step_delta_l2": info.get("final_step_delta_l2", ""),
                "patch_size": args.patch_size,
                "elapsed_sec": elapsed,
            })
            fcsv.flush()
            if args.mode in {"lir", "gated-lir"}:
                print(
                    f"[{i}/{len(files)}] mode={args.mode:10s} route={route:6s} "
                    f"plane_res_p75={info.get('plane_res_p75', '')} "
                    f"alpha={info.get('lir_alpha', '')} steps={info.get('lir_steps', '')} "
                    f"mean_delta={info.get('mean_step_delta_l2', '')} "
                    f"final_delta={info.get('final_step_delta_l2', '')} "
                    f"shape={pred.shape} out={rel(out)}",
                    flush=True,
                )
            else:
                print(
                    f"[{i}/{len(files)}] mode={args.router_mode:12s} route={route:6s} hard={info['hard_route']:6s} "
                    f"gate={float(info['gate']):.4f} plane_res_p75={info['plane_res_p75']:.6f} "
                    f"shape={pred.shape} out={rel(out)}",
                    flush=True,
                )

    print("summary")
    print(f"files: {len(files)} counts={counts} threshold={args.threshold:.6f} mode={args.mode}/{args.router_mode} soft_width={args.soft_width:.6f} lir_steps={args.lir_steps} lir_alpha={args.lir_alpha}")
    print(f"routes: {rel(route_csv)}")
    print(f"elapsed_sec: {time.time() - t_all:.1f}")
    if not args.no_zip:
        # 默认直接生成提交 zip；后续仍应跑 scripts/check_submission.py 做正式校验。
        make_zip(out_dir, zip_path)
    if args.append_registry:
        run_registry(args, zip_path, args.mode)


if __name__ == "__main__":
    main()
