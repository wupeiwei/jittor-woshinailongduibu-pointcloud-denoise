#!/usr/bin/env python3
"""
Minimal Jittor point-cloud denoising baseline for the formal competition.

Design goals:
- Keep official starter_code untouched; reuse its FeatureExtraction/Decoder when useful.
- Start with residual offset regression: pred_clean = noisy + offset.
- Make PW-SENEL optional and ablation-friendly.

PW-SENEL = PeiWei Softmax Edge-aware Noise Elimination and Locking
Core idea: Softmax noise suppression + MaxPool edge locking.

ST-AAS v0 = Structure Tensor-guided Adaptive Softmax
Core idea: density-adaptive softmax smoothing + structure-tensor edge suppression.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import queue
import random
import resource
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import trimesh
import yaml


def deep_update(base, patch):
    # YAML profile 合并规则：只覆盖 profile 里显式写出的字段，其他字段沿用 base。
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base

# 复用官方 starter_code 的特征模块，但不在这里改官方 baseline 实现。
# 自写模型和官方 VM 的边界要保持清楚，避免提交链路混淆。
ROOT = Path(__file__).resolve().parent
STARTER = ROOT / "starter_code"
sys.path.insert(0, str(STARTER))

import jittor as jt
from jittor import nn

from src.model.feature import FeatureExtraction, get_knn_idx

from denoise_utils import gather_neighbors


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)


def load_obj_vertices(path: Path) -> np.ndarray:
    mesh = trimesh.load(path, force="mesh", process=False)
    if hasattr(mesh, "vertices"):
        pts = np.asarray(mesh.vertices, dtype=np.float32)
    else:
        pts = np.asarray(mesh.dump(concatenate=True).vertices, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
        raise ValueError(f"bad obj vertices: {path}")
    return pts.astype(np.float32)


def normalize_pc(pc: np.ndarray) -> np.ndarray:
    pc = pc.astype(np.float32)
    center = pc.mean(axis=0, keepdims=True)
    pc = pc - center
    scale = np.sqrt((pc ** 2).sum(axis=1)).max()
    return pc / (scale + 1e-8)


def sample_points(pc: np.ndarray, n: int) -> np.ndarray:
    replace = len(pc) < n
    idx = np.random.choice(len(pc), n, replace=replace)
    return pc[idx].astype(np.float32)


class ObjDenoiseDataset:
    """Small, direct dataset for clean ShapeNet objs.

    It samples clean vertices and synthesizes noisy input online. This is not meant
    to be the final data pipeline; it is the first controllable baseline.
    """

    def __init__(
        self,
        data_root: str,
        list_file: str,
        num_points: int = 2048,
        noise_min: float = 0.005,
        noise_max: float = 0.02,
        limit: int = 0,
        cache_clean: bool = False,
    ):
        self.data_root = Path(data_root)
        self.num_points = num_points
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.cache_clean = cache_clean
        self._clean_cache: dict[int, np.ndarray] = {}
        # datalist 可能写 shapenet/...，也可能只写 category/model_id；下面兼容两种格式。
        ids = [x.strip() for x in Path(list_file).read_text().splitlines() if x.strip()]
        if limit > 0:
            ids = ids[:limit]
        self.files = []
        for x in ids:
            rel = Path(x)
            # Official datalist entries already start with "shapenet/...".
            if rel.parts and rel.parts[0] == "shapenet":
                p = self.data_root / rel / "models" / "model_normalized.obj"
            else:
                p = self.data_root / "shapenet" / rel / "models" / "model_normalized.obj"
            if p.exists():
                self.files.append(p)
        if not self.files:
            raise RuntimeError(f"no obj files found from {list_file} under {data_root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        # cache_clean 只缓存归一化后的 clean vertices；每次仍重新采样和加噪，
        # 所以不会把训练退化成固定 noisy/clean 对。
        if self.cache_clean and idx in self._clean_cache:
            clean_full = self._clean_cache[idx]
        else:
            clean_full = load_obj_vertices(self.files[idx])
            clean_full = normalize_pc(clean_full)
            if self.cache_clean:
                self._clean_cache[idx] = clean_full
        clean = sample_points(clean_full, self.num_points)
        sigma = np.random.uniform(self.noise_min, self.noise_max)
        noisy = clean + np.random.normal(0.0, sigma, clean.shape).astype(np.float32)
        return noisy.astype(np.float32), clean.astype(np.float32)


def scalar(x: jt.Var) -> float:
    """Best-effort scalar materialization for logging only.

    Some Jittor CUDA builds materialize scalar tensors through CuPy for all of
    `.data`, `.numpy()`, and `.item()`. CuPy is useful but should not be a hard
    dependency that kills training, so logging falls back to NaN when CuPy is not
    installed. Install `cupy-cuda12x` on CUDA 12 machines if exact live scalar
    logs are required.
    """
    try:
        return float(x.item())
    except ModuleNotFoundError as e:
        if e.name == "cupy":
            return float("nan")
        raise


def make_batch(ds: ObjDenoiseDataset, batch_size: int) -> Tuple[jt.Var, jt.Var]:
    noisy, clean = [], []
    for _ in range(batch_size):
        n, c = ds[random.randrange(len(ds))]
        noisy.append(n)
        clean.append(c)
    return jt.array(np.stack(noisy)), jt.array(np.stack(clean))


class BatchPrefetcher:
    """Prepare NumPy batches in background threads; keep Jittor work on main thread."""

    def __init__(self, ds: ObjDenoiseDataset, batch_size: int, workers: int, queue_size: int):
        self.ds = ds
        self.batch_size = batch_size
        self._stop = threading.Event()
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, queue_size))
        self._threads = []
        for i in range(max(1, workers)):
            t = threading.Thread(target=self._worker, name=f"batch-prefetch-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def _make_numpy_batch(self) -> tuple[np.ndarray, np.ndarray]:
        noisy, clean = [], []
        for _ in range(self.batch_size):
            n, c = self.ds[random.randrange(len(self.ds))]
            noisy.append(n)
            clean.append(c)
        return np.stack(noisy).astype(np.float32), np.stack(clean).astype(np.float32)

    def _worker(self) -> None:
        # 后台线程只准备 NumPy batch，不碰 Jittor 张量，降低 CUDA/Jittor 线程风险。
        while not self._stop.is_set():
            try:
                self._queue.put(self._make_numpy_batch(), timeout=0.2)
            except queue.Full:
                continue
            except Exception as e:
                try:
                    self._queue.put(e, timeout=0.2)
                except queue.Full:
                    pass

    def next_batch(self) -> Tuple[jt.Var, jt.Var]:
        item = self._queue.get()
        if isinstance(item, Exception):
            raise item
        noisy_np, clean_np = item
        return jt.array(noisy_np), jt.array(clean_np)

    def close(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=0.5)


def make_batch_prefetched(prefetcher: BatchPrefetcher | None, ds: ObjDenoiseDataset, batch_size: int) -> Tuple[jt.Var, jt.Var]:
    if prefetcher is not None:
        return prefetcher.next_batch()
    return make_batch(ds, batch_size)


def gpu_utilization() -> str:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2,
        ).strip()
        return out.replace("\n", ";")
    except Exception:
        return ""


def cpu_utilization() -> str:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    try:
        out = subprocess.check_output(
            ["ps", "-o", "%cpu,%mem,rss=", "-p", str(os.getpid())],
            text=True,
            timeout=2,
        ).strip().splitlines()
        line = out[-1].strip() if out else ""
    except Exception:
        line = ""
    return f"{line};utime={usage.ru_utime:.1f};stime={usage.ru_stime:.1f}"


class PWSENEL(nn.Module):
    """PW-SENEL: Softmax noise elimination + MaxPool edge locking.

    Given per-point features and point coordinates:
    - Softmax branch learns neighbor confidence and suppresses suspicious noisy responses.
    - MaxPool branch keeps high-response local edge/geometry cues.
    - The fused output is residual-added to original features for safe ablation.
    """

    def __init__(self, channels: int, k: int = 16):
        super().__init__()
        self.k = k
        self.score = nn.Sequential(
            nn.Linear(channels * 2 + 3, channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, 1),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(channels * 2 + 3, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
            nn.ReLU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(channels * 3, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
        )

    def gather_neighbors(self, x: jt.Var, idx: jt.Var) -> jt.Var:
        # Thin wrapper around the shared utility for backward compatibility.
        return gather_neighbors(x, idx)

    def execute(self, feat: jt.Var, points: jt.Var) -> jt.Var:
        B, N, C = feat.shape
        idx = get_knn_idx(points, points, self.k + 1)[:, :, 1:]
        neigh_feat = self.gather_neighbors(feat, idx)
        neigh_pts = self.gather_neighbors(points, idx)

        center_feat = feat.unsqueeze(2).broadcast((B, N, self.k, C))
        center_pts = points.unsqueeze(2).broadcast((B, N, self.k, 3))
        edge_input = jt.concat([center_feat, neigh_feat - center_feat, neigh_pts - center_pts], dim=-1)

        score = self.score(edge_input.reshape(B * N * self.k, -1)).reshape(B, N, self.k, 1)
        weight = nn.softmax(score, dim=2)
        soft_branch = (weight * neigh_feat).sum(dim=2)

        edge_feat = self.edge_mlp(edge_input.reshape(B * N * self.k, -1)).reshape(B, N, self.k, C)
        max_branch = jt.max(edge_feat, dim=2)

        fused = self.fuse(jt.concat([feat, soft_branch, max_branch], dim=-1).reshape(B * N, -1)).reshape(B, N, C)
        return feat + fused


class PWSENELv2Gate(nn.Module):
    """PW-SENEL v2 gate: explicit noise confidence and edge locking.

    It keeps the original offset head unchanged and only gates its movement:
    move_gate = noise_conf * (1 - edge_lock_strength * edge_conf)
    where:
    - noise_conf learns which points deserve stronger denoising;
    - edge_conf uses MaxPool-style local geometry responses to lock sharp structures.
    """

    def __init__(self, channels: int, k: int = 16, edge_lock_strength: float = 0.7, gate_scale: float = 0.5):
        super().__init__()
        self.k = k
        self.edge_lock_strength = edge_lock_strength
        self.gate_scale = gate_scale
        self.noise_mlp = nn.Sequential(
            nn.Linear(channels * 2 + 4, channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, 1),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(channels * 2 + 4, channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, 1),
        )

    def gather_neighbors(self, x: jt.Var, idx: jt.Var) -> jt.Var:
        # Thin wrapper around the shared utility for backward compatibility.
        return gather_neighbors(x, idx)

    def execute(self, feat: jt.Var, points: jt.Var, return_stats: bool = False):
        B, N, C = feat.shape
        idx = get_knn_idx(points, points, self.k + 1)[:, :, 1:]
        neigh_feat = self.gather_neighbors(feat, idx)
        neigh_pts = self.gather_neighbors(points, idx)
        center_feat = feat.unsqueeze(2).broadcast((B, N, self.k, C))
        center_pts = points.unsqueeze(2).broadcast((B, N, self.k, 3))
        rel = neigh_pts - center_pts
        dist = ((rel ** 2).sum(dim=-1, keepdims=True) + 1e-12) ** 0.5
        gate_input = jt.concat([center_feat, neigh_feat - center_feat, rel, dist], dim=-1)
        flat = gate_input.reshape(B * N * self.k, -1)

        noise_score = self.noise_mlp(flat).reshape(B, N, self.k, 1)
        noise_weight = nn.softmax(noise_score, dim=2)
        noise_conf = (noise_weight * dist).sum(dim=2)
        noise_conf = jt.sigmoid(noise_conf / (dist.mean() + 1e-6))

        edge_score = self.edge_mlp(flat).reshape(B, N, self.k, 1)
        edge_conf = jt.sigmoid(jt.max(edge_score, dim=2))

        move_gate = self.gate_scale * noise_conf * (1.0 - self.edge_lock_strength * edge_conf)
        move_gate = jt.clamp(move_gate, 0.0, 1.0)
        if return_stats:
            return move_gate, {"noise_conf": noise_conf, "edge_conf": edge_conf, "move_gate": move_gate}
        return move_gate


class STAASv0(nn.Module):
    """ST-AAS v0: Structure Tensor-guided Adaptive Softmax.

    Minimal, switchable geometry operator for ablation:
    - single KNN neighborhood;
    - local scale / density-adaptive softmax temperature;
    - structure-tensor invariant descriptors (no eigensolver dependency);
    - edge-aware smoothing suppression.

    This v0 deliberately avoids the full anisotropic matrix. It is a safe
    training-free residual branch that can be added to the neural offset output.
    """

    def __init__(
        self,
        k: int = 16,
        tau0: float = 0.02,
        tau_min: float = 0.005,
        tau_max: float = 0.08,
        density_min: float = 0.5,
        density_max: float = 2.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.k = k
        self.tau0 = tau0
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.density_min = density_min
        self.density_max = density_max
        self.eps = eps

    def gather_neighbors(self, x: jt.Var, idx: jt.Var) -> jt.Var:
        # Thin wrapper around the shared utility for backward compatibility.
        return gather_neighbors(x, idx)

    def execute(self, points: jt.Var, return_stats: bool = False):
        B, N, _ = points.shape
        idx = get_knn_idx(points, points, self.k + 1)[:, :, 1:]
        neigh = self.gather_neighbors(points, idx)
        center = points.unsqueeze(2).broadcast((B, N, self.k, 3))
        rel = neigh - center

        # Density-adaptive temperature from local scale.
        dist = ((rel ** 2).sum(dim=-1) + self.eps) ** 0.5
        sorted_dist, _ = jt.sort(dist, dim=2)
        scale = sorted_dist[:, :, self.k // 2]
        scale_avg = scale.mean(dim=1).unsqueeze(1)
        density_ratio = jt.clamp(scale_avg / (scale + self.eps), self.density_min, self.density_max)
        tau = jt.clamp(self.tau0 / ((density_ratio + self.eps) ** 0.5), self.tau_min, self.tau_max)

        # Isotropic adaptive softmax candidate for v0.
        logits = -((rel ** 2).sum(dim=-1)) / (tau.unsqueeze(-1) + self.eps)
        weight = nn.softmax(logits, dim=2).unsqueeze(-1)
        smooth = (weight * neigh).sum(dim=2)
        smooth_offset = smooth - points

        # Structure tensor / covariance invariant descriptors.
        # Avoid jt.linalg.eigh here: the A6000 competition env has historically
        # missed libcusolver.so.11, and eigensolver dependency makes smoke fail
        # before the model can train. These invariants are cheaper and stable:
        #   anisotropy: off-diagonal / diagonal covariance energy
        #   planarity: xy/xz/yz area energy relative to total variance
        #   scattering: isotropic variance balance proxy
        centered = rel - rel.mean(dim=2).unsqueeze(2)
        cov = jt.matmul(centered.permute(0, 1, 3, 2), centered) / float(self.k)
        cxx = jt.abs(cov[:, :, 0, 0])
        cyy = jt.abs(cov[:, :, 1, 1])
        czz = jt.abs(cov[:, :, 2, 2])
        cxy = cov[:, :, 0, 1]
        cxz = cov[:, :, 0, 2]
        cyz = cov[:, :, 1, 2]
        trace = cxx + cyy + czz + self.eps
        diag_min = jt.minimum(jt.minimum(cxx, cyy), czz)
        diag_max = jt.maximum(jt.maximum(cxx, cyy), czz)
        off_energy = cxy * cxy + cxz * cxz + cyz * cyz
        diag_energy = cxx * cxx + cyy * cyy + czz * czz + self.eps
        linearity = jt.clamp(off_energy / (diag_energy + off_energy + self.eps), 0.0, 1.0)
        area_energy = cxx * cyy + cxx * czz + cyy * czz
        planarity = jt.clamp(area_energy / (trace * trace + self.eps), 0.0, 1.0)
        scattering = jt.clamp(diag_min / (diag_max + self.eps), 0.0, 1.0)

        # Edge-like confidence: prefer anisotropic neighborhoods, suppress unstable scatter.
        # Planarity is tracked for logs/future v1 but not treated as edge by itself.
        edge_conf = jt.clamp(linearity * (1.0 - scattering), 0.0, 1.0)
        pred = points + (1.0 - edge_conf).unsqueeze(-1) * smooth_offset
        if return_stats:
            return pred, {
                "scale": scale,
                "tau": tau,
                "linearity": linearity,
                "planarity": planarity,
                "scattering": scattering,
                "edge_conf": edge_conf,
                "smooth_offset": smooth_offset,
            }
        return pred


class ResidualDenoiser(nn.Module):
    def __init__(
        self,
        k: int = 16,
        feat_dim: int = 256,
        hidden: int = 256,
        use_pwsenel: bool = False,
        use_staas: bool = False,
        staas_strength: float = 1.0,
        staas_tau0: float = 0.02,
        staas_tau_min: float = 0.005,
        staas_tau_max: float = 0.08,
        staas_fusion: bool = False,
        staas_v2_gate: bool = False,
        staas_v2_geo_weight: float = 0.25,
        staas_v2_gate_min: float = 0.0,
        staas_v2_gate_max: float = 1.0,
        staas_v2_noise_ref_low: float = 0.010,
        staas_v2_noise_ref_high: float = 0.030,
        use_move_gate: bool = False,
        use_pwsenel_v2: bool = False,
        pwsenel_v2_edge_lock: float = 0.7,
        pwsenel_v2_gate_scale: float = 0.5,
        residual_clip: float = 0.0,
        adaptive_clip: bool = False,
        adaptive_clip_min: float = 0.006,
        adaptive_clip_max: float = 0.020,
        adaptive_clip_ref_low: float = 0.022,
        adaptive_clip_ref_mid: float = 0.030,
        adaptive_clip_ref_high: float = 0.040,
        adaptive_clip_mid: float = 0.010,
        noise_aware_move_gate: bool = False,
        noise_aware_gate_min: float = 0.45,
        noise_aware_gate_ref_low: float = 0.022,
        noise_aware_gate_ref_high: float = 0.036,
        hybrid_safe_strong: bool = False,
        hybrid_router_scale: float = 1.0,
    ):
        super().__init__()
        # 主干编码器复用官方 FeatureExtraction；后面的 head/gate 是本仓库研究层。
        self.encoder = FeatureExtraction(k=k, input_dim=3, embedding_dim=feat_dim)
        self.use_pwsenel = use_pwsenel
        self.pwsenel = PWSENEL(feat_dim, k=k) if use_pwsenel else None
        self.use_pwsenel_v2 = use_pwsenel_v2
        self.pwsenel_v2_gate = PWSENELv2Gate(
            feat_dim,
            k=k,
            edge_lock_strength=pwsenel_v2_edge_lock,
            gate_scale=pwsenel_v2_gate_scale,
        ) if use_pwsenel_v2 else None
        self.use_staas = use_staas
        self.staas_strength = staas_strength
        self.staas_fusion = staas_fusion
        self.staas_v2_gate = staas_v2_gate
        self.staas_v2_geo_weight = staas_v2_geo_weight
        self.staas_v2_gate_min = staas_v2_gate_min
        self.staas_v2_gate_max = staas_v2_gate_max
        self.staas_v2_noise_ref_low = staas_v2_noise_ref_low
        self.staas_v2_noise_ref_high = staas_v2_noise_ref_high
        self.residual_clip = residual_clip
        self.adaptive_clip = adaptive_clip
        self.adaptive_clip_min = adaptive_clip_min
        self.adaptive_clip_max = adaptive_clip_max
        self.adaptive_clip_ref_low = adaptive_clip_ref_low
        self.adaptive_clip_ref_mid = adaptive_clip_ref_mid
        self.adaptive_clip_ref_high = adaptive_clip_ref_high
        self.adaptive_clip_mid = adaptive_clip_mid
        self.noise_aware_move_gate = noise_aware_move_gate
        self.noise_aware_gate_min = noise_aware_gate_min
        self.noise_aware_gate_ref_low = noise_aware_gate_ref_low
        self.noise_aware_gate_ref_high = noise_aware_gate_ref_high
        self.hybrid_safe_strong = hybrid_safe_strong
        self.hybrid_router_scale = hybrid_router_scale
        self.staas = STAASv0(k=k, tau0=staas_tau0, tau_min=staas_tau_min, tau_max=staas_tau_max) if (use_staas or staas_fusion or staas_v2_gate) else None
        self.staas_geo_fuse = nn.Sequential(
            nn.Linear(feat_dim + 9, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feat_dim),
            nn.ReLU(),
        ) if staas_fusion else None
        self.staas_v2_gate_net = nn.Sequential(
            nn.Linear(feat_dim + 9, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        ) if staas_v2_gate else None
        self.staas_v2_geo_gate_net = nn.Sequential(
            nn.Linear(feat_dim + 9, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        ) if staas_v2_gate else None
        self.k_for_scale = k
        self.use_move_gate = use_move_gate
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )
        self.strong_head = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        ) if hybrid_safe_strong else None
        self.move_gate = nn.Sequential(
            nn.Linear(feat_dim, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        ) if use_move_gate else None

    def _gather_neighbors(self, x: jt.Var, idx: jt.Var) -> jt.Var:
        # Thin wrapper around the shared utility for backward compatibility.
        return gather_neighbors(x, idx)

    def execute(self, noisy: jt.Var, return_offset: bool = False):
        # 主路径：noisy -> feature -> residual offset -> 可选门控/裁剪/几何分支 -> pred。
        # 所有研究模块都以开关形式挂在这条路径上，便于做 ablation 和回退。
        feat = self.encoder(noisy)
        if self.pwsenel is not None:
            feat = self.pwsenel(feat, noisy)
        B, N, C = feat.shape
        staas_pred = None
        staas_stats = None
        if self.staas is not None and (self.staas_fusion or self.staas_v2_gate):
            staas_pred, staas_stats = self.staas(noisy, return_stats=True)
            geo = jt.concat([
                staas_stats["smooth_offset"],
                staas_stats["scale"].unsqueeze(-1),
                staas_stats["tau"].unsqueeze(-1),
                staas_stats["linearity"].unsqueeze(-1),
                staas_stats["planarity"].unsqueeze(-1),
                staas_stats["scattering"].unsqueeze(-1),
                staas_stats["edge_conf"].unsqueeze(-1),
            ], dim=-1)
            if self.staas_fusion:
                feat = self.staas_geo_fuse(jt.concat([feat, geo], dim=-1).reshape(B * N, C + 9)).reshape(B, N, C)
        neural_offset = self.head(feat.reshape(B * N, C)).reshape(B, N, 3)

        if self.hybrid_safe_strong:
            # Hybrid safe/strong denoising:
            # - safe branch is protected by PW-SENEL v2 and conservative clipping;
            # - strong branch keeps move_gate capacity for high-noise clouds;
            # - router opens the strong branch only when local noise is high and
            #   edge confidence is low, further modulated by per-cloud scale.
            idx = get_knn_idx(noisy, noisy, self.k_for_scale + 1)[:, :, 1:]
            neigh = self._gather_neighbors(noisy, idx)
            local_dist = (((neigh - noisy.unsqueeze(2)) ** 2).sum(dim=-1) + 1e-12) ** 0.5
            cloud_scale = local_dist.mean(dim=2, keepdims=True).mean(dim=1, keepdims=True)

            if self.pwsenel_v2_gate is not None:
                safe_gate, stats = self.pwsenel_v2_gate(feat, noisy, return_stats=True)
                noise_conf = stats["noise_conf"]
                edge_conf = stats["edge_conf"]
            else:
                safe_gate = 1.0
                mean_dist = local_dist.mean(dim=2, keepdims=True)
                noise_conf = jt.sigmoid(mean_dist / (local_dist.mean() + 1e-6))
                edge_conf = 0.0

            safe_offset = neural_offset * safe_gate

            strong_offset = self.strong_head(feat.reshape(B * N, C)).reshape(B, N, 3)
            if self.move_gate is not None:
                strong_gate = self.move_gate(feat.reshape(B * N, C)).reshape(B, N, 1)
                strong_offset = strong_offset * strong_gate

            t_cloud = jt.clamp((cloud_scale - self.adaptive_clip_ref_low) / (self.adaptive_clip_ref_high - self.adaptive_clip_ref_low + 1e-12), 0.0, 1.0)
            router = jt.clamp(self.hybrid_router_scale * noise_conf * (1.0 - edge_conf) * t_cloud, 0.0, 1.0)
            offset = safe_offset * (1.0 - router) + strong_offset * router
        else:
            if self.move_gate is not None:
                gate = self.move_gate(feat.reshape(B * N, C)).reshape(B, N, 1)
                if self.noise_aware_move_gate:
                    # Suppress learned movement on low-noise clouds where global
                    # move_gate previously over-corrected, while leaving mid/high
                    # noise clouds close to the original move_gate behavior.
                    idx = get_knn_idx(noisy, noisy, self.k_for_scale + 1)[:, :, 1:]
                    neigh = self._gather_neighbors(noisy, idx)
                    local_dist = (((neigh - noisy.unsqueeze(2)) ** 2).sum(dim=-1) + 1e-12) ** 0.5
                    cloud_scale = local_dist.mean(dim=2, keepdims=True).mean(dim=1, keepdims=True)
                    t = jt.clamp((cloud_scale - self.noise_aware_gate_ref_low) / (self.noise_aware_gate_ref_high - self.noise_aware_gate_ref_low + 1e-12), 0.0, 1.0)
                    cloud_gate = self.noise_aware_gate_min + t * (1.0 - self.noise_aware_gate_min)
                    gate = gate * cloud_gate
                neural_offset = neural_offset * gate
            if self.pwsenel_v2_gate is not None:
                gate = self.pwsenel_v2_gate(feat, noisy)
                neural_offset = neural_offset * gate
            offset = neural_offset
        if self.staas_v2_gate and staas_stats is not None:
            # ST-AAS v2: noisy-conditioned residual gate. The first v2 smoke
            # showed a common failure mode: multiplying the whole neural offset
            # by a learned/noise/edge gate made the model too timid at mid/high
            # noise, where the baseline wins by moving decisively. So the gate
            # is intentionally one-sided now: protect low-noise/edge regions via
            # a lower-bound schedule, but do not suppress high-noise residuals
            # below the neural head's own prediction. Geometry remains a small
            # auxiliary offset and gets its own gate.
            gate_in = jt.concat([feat, geo], dim=-1).reshape(B * N, C + 9)
            learned_protect = self.staas_v2_gate_net(gate_in).reshape(B, N, 1)
            learned_geo = self.staas_v2_geo_gate_net(gate_in).reshape(B, N, 1)
            noise_t = jt.clamp(
                (staas_stats["scale"].unsqueeze(-1) - self.staas_v2_noise_ref_low)
                / (self.staas_v2_noise_ref_high - self.staas_v2_noise_ref_low + 1e-12),
                0.0,
                1.0,
            )
            flat_lock = 1.0 - staas_stats["edge_conf"].unsqueeze(-1)
            # protect_t is high only for low-noise / edge-like neighborhoods.
            protect_t = jt.clamp((1.0 - noise_t) + staas_stats["edge_conf"].unsqueeze(-1), 0.0, 1.0)
            residual_gate = 1.0 - (1.0 - self.staas_v2_gate_min) * learned_protect * protect_t
            residual_gate = jt.clamp(residual_gate, self.staas_v2_gate_min, self.staas_v2_gate_max)
            geo_offset = staas_stats["smooth_offset"].stop_grad()
            geo_gate = learned_geo * noise_t * flat_lock
            offset = offset * residual_gate + self.staas_v2_geo_weight * geo_offset * geo_gate
        if self.adaptive_clip:
            # Piecewise cloud-scale adaptive clipping. Keep low-noise clouds
            # conservative, reach a v2_clip-like mid residual around ref_mid,
            # then gradually open high-noise clouds without the low-noise damage
            # seen from a single aggressive linear ramp.
            idx = get_knn_idx(noisy, noisy, self.pwsenel_v2_gate.k + 1 if self.pwsenel_v2_gate is not None else 17)[:, :, 1:]
            neigh = PWSENELv2Gate.gather_neighbors(self.pwsenel_v2_gate, noisy, idx) if self.pwsenel_v2_gate is not None else self._gather_neighbors(noisy, idx)
            local_dist = (((neigh - noisy.unsqueeze(2)) ** 2).sum(dim=-1) + 1e-12) ** 0.5
            cloud_scale = local_dist.mean(dim=2, keepdims=True).mean(dim=1, keepdims=True)
            t_low = jt.clamp((cloud_scale - self.adaptive_clip_ref_low) / (self.adaptive_clip_ref_mid - self.adaptive_clip_ref_low + 1e-12), 0.0, 1.0)
            t_high = jt.clamp((cloud_scale - self.adaptive_clip_ref_mid) / (self.adaptive_clip_ref_high - self.adaptive_clip_ref_mid + 1e-12), 0.0, 1.0)
            clip_low = self.adaptive_clip_min + t_low * (self.adaptive_clip_mid - self.adaptive_clip_min)
            clip_high = self.adaptive_clip_mid + t_high * (self.adaptive_clip_max - self.adaptive_clip_mid)
            high_mask = (cloud_scale > self.adaptive_clip_ref_mid).float32()
            clip = clip_low * (1.0 - high_mask) + clip_high * high_mask
            offset_norm = ((offset ** 2).sum(dim=-1, keepdims=True) + 1e-12) ** 0.5
            scale = jt.clamp(clip / (offset_norm + 1e-12), 0.0, 1.0)
            offset = offset * scale
        elif self.residual_clip and self.residual_clip > 0:
            offset_norm = ((offset ** 2).sum(dim=-1, keepdims=True) + 1e-12) ** 0.5
            scale = jt.clamp(self.residual_clip / (offset_norm + 1e-12), 0.0, 1.0)
            offset = offset * scale
        if self.staas is not None and self.use_staas:
            if staas_pred is None:
                staas_pred = self.staas(noisy)
            staas_offset = (staas_pred - noisy).stop_grad()
            offset = offset + self.staas_strength * staas_offset
        pred = noisy + offset
        if return_offset:
            return pred, offset
        return pred


def chamfer_l2(a: jt.Var, b: jt.Var) -> jt.Var:
    # simple dense CD for small training batches; patch/fast metric can replace later.
    dist = ((a.unsqueeze(2) - b.unsqueeze(1)) ** 2).sum(dim=-1)
    return jt.min(dist, dim=2).mean() + jt.min(dist, dim=1).mean()


def pointwise_offset_loss(pred: jt.Var, clean: jt.Var, loss_type: str = "mse", huber_delta: float = 0.01) -> jt.Var:
    """Point-wise offset regression loss with a default-preserving switch.

    ``mse`` keeps the historical baseline behavior. ``huber`` / ``smooth_l1``
    are low-risk recipe knobs for robust offset training; they are opt-in and
    should be ablated before any official submission candidate is generated.
    """
    diff = pred - clean
    if loss_type == "mse":
        return (diff ** 2).mean()
    abs_diff = jt.abs(diff)
    delta = float(huber_delta)
    if delta <= 0:
        raise ValueError(f"huber_delta must be positive, got {huber_delta}")
    if loss_type == "huber":
        quadratic = jt.minimum(abs_diff, delta)
        linear = abs_diff - quadratic
        return (0.5 * quadratic ** 2 + delta * linear).mean()
    if loss_type == "smooth_l1":
        quadratic = jt.minimum(abs_diff, delta)
        linear = abs_diff - quadratic
        return (0.5 * quadratic ** 2 / delta + linear).mean()
    raise ValueError(f"unknown loss_type: {loss_type}")


def lr_factor_for_step(step: int, total_steps: int, scheduler: str = "none", warmup_steps: int = 0, eta_min_ratio: float = 0.05) -> float:
    """Return a multiplicative LR factor for train-step schedulers.

    Default ``scheduler=none`` and ``warmup_steps=0`` exactly preserves the old
    constant-LR behavior. Warmup is linear; cosine decay starts after warmup.
    """
    scheduler = (scheduler or "none").lower()
    if scheduler not in {"none", "constant", "cosine", "warmup_cosine"}:
        raise ValueError(f"unknown lr_scheduler: {scheduler}")
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(1, int(total_steps))
    eta_min_ratio = float(eta_min_ratio)
    if warmup_steps > 0 and step <= warmup_steps:
        return max(float(step) / float(warmup_steps), 1.0 / float(warmup_steps))
    if scheduler in {"none", "constant"}:
        return 1.0
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
    return eta_min_ratio + (1.0 - eta_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def check_paths(args, need_train: bool = False, need_test: bool = False, need_ckpt: bool = False) -> None:
    if need_train:
        if not Path(args.data_root).exists():
            raise FileNotFoundError(f"data_root not found: {args.data_root}")
        if not Path(args.train_list).exists():
            raise FileNotFoundError(f"train_list not found: {args.train_list}")
    if need_test and not Path(args.test_root).exists():
        raise FileNotFoundError(f"test_root not found: {args.test_root}")
    if need_ckpt and not Path(args.ckpt).exists():
        raise FileNotFoundError(f"ckpt not found: {args.ckpt}")


def write_run_summary(args, ds_len: int = 0) -> None:
    ckpt_dir = Path(args.ckpt).parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    summary = ckpt_dir / "last_run_summary.txt"
    lines = [
        f"time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"experiment_name: {args.experiment_name}",
        f"config: {args.config}",
        f"profiles: {','.join(args.profile) if args.profile else 'none'}",
        f"data_root: {args.data_root}",
        f"train_list: {args.train_list}",
        f"dataset_size: {ds_len}",
        f"ckpt: {args.ckpt}",
        f"steps: {args.steps}",
        f"batch_size: {args.batch_size}",
        f"num_points: {args.num_points}",
        f"k: {args.k}",
        f"feat_dim: {args.feat_dim}",
        f"hidden: {args.hidden}",
        f"lr: {args.lr}",
        f"loss_type: {args.loss_type}",
        f"huber_delta: {args.huber_delta}",
        f"lr_scheduler: {args.lr_scheduler}",
        f"warmup_steps: {args.warmup_steps}",
        f"eta_min_ratio: {args.eta_min_ratio}",
        f"pwsenel: {args.pwsenel}",
        f"staas: {args.staas}",
        f"staas_strength: {args.staas_strength}",
        f"staas_tau0: {args.staas_tau0}",
        f"staas_tau_min: {args.staas_tau_min}",
        f"staas_tau_max: {args.staas_tau_max}",
        f"staas_fusion: {args.staas_fusion}",
        f"staas_v2_gate: {args.staas_v2_gate}",
        f"staas_v2_geo_weight: {args.staas_v2_geo_weight}",
        f"staas_v2_gate_min: {args.staas_v2_gate_min}",
        f"staas_v2_gate_max: {args.staas_v2_gate_max}",
        f"staas_v2_noise_ref_low: {args.staas_v2_noise_ref_low}",
        f"staas_v2_noise_ref_high: {args.staas_v2_noise_ref_high}",
        f"move_gate: {args.move_gate}",
        f"pwsenel_v2: {args.pwsenel_v2}",
        f"pwsenel_v2_edge_lock: {args.pwsenel_v2_edge_lock}",
        f"pwsenel_v2_gate_scale: {args.pwsenel_v2_gate_scale}",
        f"residual_clip: {args.residual_clip}",
        f"adaptive_clip: {args.adaptive_clip}",
        f"adaptive_clip_min: {args.adaptive_clip_min}",
        f"adaptive_clip_max: {args.adaptive_clip_max}",
        f"adaptive_clip_ref_low: {args.adaptive_clip_ref_low}",
        f"adaptive_clip_ref_mid: {args.adaptive_clip_ref_mid}",
        f"adaptive_clip_ref_high: {args.adaptive_clip_ref_high}",
        f"adaptive_clip_mid: {args.adaptive_clip_mid}",
        f"noise_aware_move_gate: {args.noise_aware_move_gate}",
        f"noise_aware_gate_min: {args.noise_aware_gate_min}",
        f"noise_aware_gate_ref_low: {args.noise_aware_gate_ref_low}",
        f"noise_aware_gate_ref_high: {args.noise_aware_gate_ref_high}",
        f"cache_clean: {args.cache_clean}",
        f"prefetch_workers: {args.prefetch_workers}",
        f"prefetch_queue_size: {args.prefetch_queue_size}",
        f"profile_times: {args.profile_times}",
        f"profile_system_every: {args.profile_system_every}",
        f"hybrid_safe_strong: {args.hybrid_safe_strong}",
        f"hybrid_router_scale: {args.hybrid_router_scale}",
    ]
    summary.write_text("\n".join(lines) + "\n")


def train(args) -> None:
    # 训练入口只负责自写 ResidualDenoiser，不训练官方 starter_code VM。
    check_paths(args, need_train=True)
    ds = ObjDenoiseDataset(
        data_root=args.data_root,
        list_file=args.train_list,
        num_points=args.num_points,
        noise_min=args.noise_min,
        noise_max=args.noise_max,
        limit=args.limit,
        cache_clean=args.cache_clean,
    )
    model = ResidualDenoiser(
        k=args.k,
        feat_dim=args.feat_dim,
        hidden=args.hidden,
        use_pwsenel=args.pwsenel,
        use_staas=args.staas,
        staas_strength=args.staas_strength,
        staas_tau0=args.staas_tau0,
        staas_tau_min=args.staas_tau_min,
        staas_tau_max=args.staas_tau_max,
        staas_fusion=args.staas_fusion,
        staas_v2_gate=args.staas_v2_gate,
        staas_v2_geo_weight=args.staas_v2_geo_weight,
        staas_v2_gate_min=args.staas_v2_gate_min,
        staas_v2_gate_max=args.staas_v2_gate_max,
        staas_v2_noise_ref_low=args.staas_v2_noise_ref_low,
        staas_v2_noise_ref_high=args.staas_v2_noise_ref_high,
        use_move_gate=args.move_gate,
        use_pwsenel_v2=args.pwsenel_v2,
        pwsenel_v2_edge_lock=args.pwsenel_v2_edge_lock,
        pwsenel_v2_gate_scale=args.pwsenel_v2_gate_scale,
        residual_clip=args.residual_clip,
        adaptive_clip=args.adaptive_clip,
        adaptive_clip_min=args.adaptive_clip_min,
        adaptive_clip_max=args.adaptive_clip_max,
        adaptive_clip_ref_low=args.adaptive_clip_ref_low,
        adaptive_clip_ref_mid=args.adaptive_clip_ref_mid,
        adaptive_clip_ref_high=args.adaptive_clip_ref_high,
        adaptive_clip_mid=args.adaptive_clip_mid,
        noise_aware_move_gate=args.noise_aware_move_gate,
        noise_aware_gate_min=args.noise_aware_gate_min,
        noise_aware_gate_ref_low=args.noise_aware_gate_ref_low,
        noise_aware_gate_ref_high=args.noise_aware_gate_ref_high,
        hybrid_safe_strong=args.hybrid_safe_strong,
        hybrid_router_scale=args.hybrid_router_scale,
    )
    os.makedirs(Path(args.ckpt).parent, exist_ok=True)
    warm_start = getattr(args, "warm_start", "")
    if warm_start:
        print(f"warm_start: loading compatible params from {warm_start}", flush=True)
        model.load(warm_start)
    opt = nn.Adam(model.parameters(), lr=args.lr)
    prefetcher = None
    if args.prefetch_workers > 0:
        prefetcher = BatchPrefetcher(
            ds,
            batch_size=args.batch_size,
            workers=args.prefetch_workers,
            queue_size=args.prefetch_queue_size,
        )
        print(
            f"input prefetch enabled: workers={args.prefetch_workers} "
            f"queue_size={args.prefetch_queue_size}",
            flush=True,
        )
    os.makedirs(Path(args.ckpt).parent, exist_ok=True)
    write_run_summary(args, ds_len=len(ds))
    csv_fields = [
        "step",
        "lr",
        "loss",
        "offset_mse",
        "cd",
        "pred_offset_mean",
        "pred_offset_abs_mean",
        "pred_offset_l2_mean",
        "identity_loss",
        "movement_loss",
        "data_time_sec",
        "compute_time_sec",
        "step_time_sec",
        "elapsed_sec",
    ]
    csv_path = Path(args.ckpt).with_suffix(".train.csv")
    # Avoid appending new-schema rows to old logs whose column was named `offset`
    # even though it actually stored MSE(pred, clean).
    if csv_path.exists():
        first_line = csv_path.open("r", newline="").readline().strip()
        if first_line and first_line != ",".join(csv_fields):
            csv_path = csv_path.with_name(csv_path.stem + ".v2" + csv_path.suffix)
    csv_new = not csv_path.exists()
    csv_f = csv_path.open("a", newline="")
    csv_w = csv.DictWriter(csv_f, fieldnames=csv_fields)
    if csv_new:
        csv_w.writeheader()
    t0 = time.time()

    prev_step_end = time.time()
    try:
        for step in range(1, args.steps + 1):
            # data_time 和 compute_time 分开记，便于判断瓶颈在 OBJ/NumPy 输入还是 Jittor 前后向。
            step_start = time.time()
            noisy, clean = make_batch_prefetched(prefetcher, ds, args.batch_size)
            data_ready = time.time()
            lr_factor = lr_factor_for_step(
                step,
                args.steps,
                scheduler=args.lr_scheduler,
                warmup_steps=args.warmup_steps,
                eta_min_ratio=args.eta_min_ratio,
            )
            current_lr = args.lr * lr_factor
            opt.lr = current_lr
            pred, pred_offset = model(noisy, return_offset=True)
            loss_offset = pointwise_offset_loss(pred, clean, loss_type=args.loss_type, huber_delta=args.huber_delta)
            loss_cd = chamfer_l2(pred, clean) if args.cd_weight > 0 else jt.array(0.0)
            loss_identity = ((pred - noisy) ** 2).mean() if args.identity_weight > 0 else jt.array(0.0)
            loss_movement = ((pred_offset ** 2).sum(dim=-1) + 1e-12).mean() if args.movement_weight > 0 else jt.array(0.0)
            loss = loss_offset + args.cd_weight * loss_cd + args.identity_weight * loss_identity + args.movement_weight * loss_movement
            opt.step(loss)
            compute_done = time.time()
            data_time = data_ready - step_start
            compute_time = compute_done - data_ready
            step_time = compute_done - prev_step_end
            prev_step_end = compute_done
            loss_v = scalar(loss)
            offset_mse_v = scalar(loss_offset)
            cd_v = scalar(loss_cd)
            pred_offset_mean_v = scalar(pred_offset.mean())
            pred_offset_abs_mean_v = scalar(jt.abs(pred_offset).mean())
            pred_offset_l2_mean_v = scalar((((pred_offset ** 2).sum(dim=-1) + 1e-12) ** 0.5).mean())
            identity_loss_v = scalar(loss_identity)
            movement_loss_v = scalar(loss_movement)
            if step == 1 or step % args.log_every == 0:
                elapsed = time.time() - t0
                profile_now = args.profile_times or (args.profile_system_every > 0 and step % args.profile_system_every == 0)
                profile_msg = ""
                if args.profile_times:
                    profile_msg = (
                        f" data_time={data_time:.4f}s compute_time={compute_time:.4f}s "
                        f"step_time={step_time:.4f}s"
                    )
                if profile_now:
                    profile_msg += f" gpu_util='{gpu_utilization()}' cpu_util='{cpu_utilization()}'"
                print(
                    f"step={step} lr={current_lr:.6g} loss={loss_v:.6f} offset_loss={offset_mse_v:.6f} "
                    f"cd={cd_v:.6f} pred_offset_abs_mean={pred_offset_abs_mean_v:.6f} "
                    f"pred_offset_l2_mean={pred_offset_l2_mean_v:.6f} elapsed={elapsed:.1f}s{profile_msg}",
                    flush=True,
                )
                csv_w.writerow({
                    "step": step,
                    "lr": current_lr,
                    "loss": loss_v,
                    "offset_mse": offset_mse_v,
                    "cd": cd_v,
                    "pred_offset_mean": pred_offset_mean_v,
                    "pred_offset_abs_mean": pred_offset_abs_mean_v,
                    "pred_offset_l2_mean": pred_offset_l2_mean_v,
                    "identity_loss": identity_loss_v,
                    "movement_loss": movement_loss_v,
                    "data_time_sec": data_time,
                    "compute_time_sec": compute_time,
                    "step_time_sec": step_time,
                    "elapsed_sec": elapsed,
                })
                csv_f.flush()
            if step % args.save_every == 0 or step == args.steps:
                model.save(args.ckpt)
                print(f"saved {args.ckpt}", flush=True)
    finally:
        if prefetcher is not None:
            prefetcher.close()
        csv_f.close()


def predict_points_in_chunks(model: ResidualDenoiser, noisy_np: np.ndarray, patch_size: int) -> np.ndarray:
    """Memory-safe inference for large point clouds.

    The first baseline uses independent contiguous chunks. This is intentionally
    simple; later we can replace it with FPS/KNN overlapping patches + weighted
    stitching like the official baseline.
    """
    # 这是最保守的省显存推理兜底：连续 chunk 独立预测，不做跨 chunk stitching。
    outs = []
    with jt.no_grad():
        for start in range(0, len(noisy_np), patch_size):
            chunk = noisy_np[start:start + patch_size]
            pred = model(jt.array(chunk[None, ...])).numpy()[0]
            outs.append(pred.astype(np.float32))
    return np.concatenate(outs, axis=0).astype(np.float32)


def predict(args) -> None:
    # 预测入口只做加载 checkpoint、前向推理、写 denoised.npy，不做训练或质量评估。
    check_paths(args, need_test=True, need_ckpt=True)
    model = ResidualDenoiser(
        k=args.k,
        feat_dim=args.feat_dim,
        hidden=args.hidden,
        use_pwsenel=args.pwsenel,
        use_staas=args.staas,
        staas_strength=args.staas_strength,
        staas_tau0=args.staas_tau0,
        staas_tau_min=args.staas_tau_min,
        staas_tau_max=args.staas_tau_max,
        staas_fusion=args.staas_fusion,
        staas_v2_gate=args.staas_v2_gate,
        staas_v2_geo_weight=args.staas_v2_geo_weight,
        staas_v2_gate_min=args.staas_v2_gate_min,
        staas_v2_gate_max=args.staas_v2_gate_max,
        staas_v2_noise_ref_low=args.staas_v2_noise_ref_low,
        staas_v2_noise_ref_high=args.staas_v2_noise_ref_high,
        use_move_gate=args.move_gate,
        use_pwsenel_v2=args.pwsenel_v2,
        pwsenel_v2_edge_lock=args.pwsenel_v2_edge_lock,
        pwsenel_v2_gate_scale=args.pwsenel_v2_gate_scale,
        residual_clip=args.residual_clip,
        adaptive_clip=args.adaptive_clip,
        adaptive_clip_min=args.adaptive_clip_min,
        adaptive_clip_max=args.adaptive_clip_max,
        adaptive_clip_ref_low=args.adaptive_clip_ref_low,
        adaptive_clip_ref_mid=args.adaptive_clip_ref_mid,
        adaptive_clip_ref_high=args.adaptive_clip_ref_high,
        adaptive_clip_mid=args.adaptive_clip_mid,
        noise_aware_move_gate=args.noise_aware_move_gate,
        noise_aware_gate_min=args.noise_aware_gate_min,
        noise_aware_gate_ref_low=args.noise_aware_gate_ref_low,
        noise_aware_gate_ref_high=args.noise_aware_gate_ref_high,
        hybrid_safe_strong=args.hybrid_safe_strong,
        hybrid_router_scale=args.hybrid_router_scale,
    )
    model.load(args.ckpt)
    model.eval()
    test_root = Path(args.test_root)
    out_root = Path(args.out_dir)
    files = sorted(test_root.glob("shapenet/*/*/noisy.npy"))
    if args.limit > 0:
        files = files[:args.limit]
    if not files:
        raise RuntimeError(f"no test noisy.npy files found under {test_root}")
    for i, f in enumerate(files, 1):
        noisy_np = np.load(f).astype(np.float32)
        pred = predict_points_in_chunks(model, noisy_np, args.predict_patch_size)
        rel = f.relative_to(test_root)
        out = out_root / rel.parent / "denoised.npy"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, pred)
        print(f"[{i}/{len(files)}] {out} shape={pred.shape}")


def make_zip(args) -> None:
    # 打包入口只把 out_dir 下的正式相对路径写入 zip，不重新推理。
    out_dir = Path(args.out_dir)
    if not out_dir.exists():
        raise FileNotFoundError(f"out_dir not found: {out_dir}")
    files = sorted(out_dir.glob("shapenet/*/*/denoised.npy"))
    if not files:
        raise RuntimeError(f"no denoised.npy files found under {out_dir}")
    zip_path = Path(args.zip)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, f.relative_to(out_dir))
    print(f"wrote {zip_path} files={len(files)}")


def validate_zip(args) -> None:
    # 这里是轻量路径检查；完整 shape/dtype/finite 检查应使用 scripts/check_submission.py。
    zip_path = Path(args.zip)
    if not zip_path.exists():
        raise FileNotFoundError(f"zip not found: {zip_path}")
    bad = []
    count = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            count += 1
            parts = Path(name).parts
            if len(parts) != 4 or parts[0] != "shapenet" or parts[-1] != "denoised.npy":
                bad.append(name)
    if bad:
        raise RuntimeError(f"bad zip entries: {bad[:10]}")
    print(f"zip ok: {zip_path} files={count}")


def apply_config(args):
    """Load YAML config, then re-apply explicit CLI overrides.

    argparse has no built-in way to tell defaults from user-provided values, so
    main() records CLI overrides by comparing against parser defaults before
    calling this function.
    """
    overrides = getattr(args, "_cli_overrides", {})
    if not args.config:
        return args
    # 合并顺序：config -> profiles -> 显式 CLI 覆盖项。
    # 这样 quick debug 的 `--steps 10` 不会被 YAML 默认值覆盖掉。
    cfg = yaml.safe_load(Path(args.config).read_text())
    for profile in args.profile:
        patch = yaml.safe_load(Path(profile).read_text())
        cfg = deep_update(cfg, patch)
    exp = cfg.get("experiment", {})
    paths = cfg.get("paths", {})
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    pred_cfg = cfg.get("predict", {})

    args.experiment_name = exp.get("name", args.experiment_name)
    args.seed = exp.get("seed", args.seed)

    args.data_root = paths.get("data_root", args.data_root)
    args.test_root = paths.get("test_root", args.test_root)
    args.train_list = paths.get("train_list", args.train_list)
    args.out_dir = paths.get("out_dir", args.out_dir)
    args.zip = paths.get("zip", args.zip)
    args.ckpt = paths.get("ckpt", args.ckpt)
    args.warm_start = paths.get("warm_start", args.warm_start)

    args.steps = train_cfg.get("steps", args.steps)
    args.limit = train_cfg.get("limit", args.limit)
    args.num_points = train_cfg.get("num_points", args.num_points)
    args.batch_size = train_cfg.get("batch_size", args.batch_size)
    args.lr = train_cfg.get("lr", args.lr)
    args.loss_type = train_cfg.get("loss_type", args.loss_type)
    args.huber_delta = train_cfg.get("huber_delta", args.huber_delta)
    args.lr_scheduler = train_cfg.get("lr_scheduler", args.lr_scheduler)
    args.warmup_steps = train_cfg.get("warmup_steps", args.warmup_steps)
    args.eta_min_ratio = train_cfg.get("eta_min_ratio", args.eta_min_ratio)
    args.noise_min = train_cfg.get("noise_min", args.noise_min)
    args.noise_max = train_cfg.get("noise_max", args.noise_max)
    args.cd_weight = train_cfg.get("cd_weight", args.cd_weight)
    args.identity_weight = train_cfg.get("identity_weight", args.identity_weight)
    args.movement_weight = train_cfg.get("movement_weight", args.movement_weight)
    args.log_every = train_cfg.get("log_every", args.log_every)
    args.save_every = train_cfg.get("save_every", args.save_every)

    args.k = model_cfg.get("k", args.k)
    args.feat_dim = model_cfg.get("feat_dim", args.feat_dim)
    args.hidden = model_cfg.get("hidden", args.hidden)
    args.pwsenel = model_cfg.get("pwsenel", args.pwsenel)
    args.staas = model_cfg.get("staas", args.staas)
    args.staas_strength = model_cfg.get("staas_strength", args.staas_strength)
    args.staas_tau0 = model_cfg.get("staas_tau0", args.staas_tau0)
    args.staas_tau_min = model_cfg.get("staas_tau_min", args.staas_tau_min)
    args.staas_tau_max = model_cfg.get("staas_tau_max", args.staas_tau_max)
    args.staas_fusion = model_cfg.get("staas_fusion", args.staas_fusion)
    args.staas_v2_gate = model_cfg.get("staas_v2_gate", args.staas_v2_gate)
    args.staas_v2_geo_weight = model_cfg.get("staas_v2_geo_weight", args.staas_v2_geo_weight)
    args.staas_v2_gate_min = model_cfg.get("staas_v2_gate_min", args.staas_v2_gate_min)
    args.staas_v2_gate_max = model_cfg.get("staas_v2_gate_max", args.staas_v2_gate_max)
    args.staas_v2_noise_ref_low = model_cfg.get("staas_v2_noise_ref_low", args.staas_v2_noise_ref_low)
    args.staas_v2_noise_ref_high = model_cfg.get("staas_v2_noise_ref_high", args.staas_v2_noise_ref_high)
    args.move_gate = model_cfg.get("move_gate", args.move_gate)
    args.pwsenel_v2 = model_cfg.get("pwsenel_v2", args.pwsenel_v2)
    args.pwsenel_v2_edge_lock = model_cfg.get("pwsenel_v2_edge_lock", args.pwsenel_v2_edge_lock)
    args.pwsenel_v2_gate_scale = model_cfg.get("pwsenel_v2_gate_scale", args.pwsenel_v2_gate_scale)
    args.residual_clip = model_cfg.get("residual_clip", args.residual_clip)
    args.adaptive_clip = model_cfg.get("adaptive_clip", args.adaptive_clip)
    args.adaptive_clip_min = model_cfg.get("adaptive_clip_min", args.adaptive_clip_min)
    args.adaptive_clip_max = model_cfg.get("adaptive_clip_max", args.adaptive_clip_max)
    args.adaptive_clip_ref_low = model_cfg.get("adaptive_clip_ref_low", args.adaptive_clip_ref_low)
    args.adaptive_clip_ref_mid = model_cfg.get("adaptive_clip_ref_mid", args.adaptive_clip_ref_mid)
    args.adaptive_clip_ref_high = model_cfg.get("adaptive_clip_ref_high", args.adaptive_clip_ref_high)
    args.adaptive_clip_mid = model_cfg.get("adaptive_clip_mid", args.adaptive_clip_mid)
    args.noise_aware_move_gate = model_cfg.get("noise_aware_move_gate", args.noise_aware_move_gate)
    args.noise_aware_gate_min = model_cfg.get("noise_aware_gate_min", args.noise_aware_gate_min)
    args.noise_aware_gate_ref_low = model_cfg.get("noise_aware_gate_ref_low", args.noise_aware_gate_ref_low)
    args.noise_aware_gate_ref_high = model_cfg.get("noise_aware_gate_ref_high", args.noise_aware_gate_ref_high)
    args.hybrid_safe_strong = model_cfg.get("hybrid_safe_strong", args.hybrid_safe_strong)
    args.hybrid_router_scale = model_cfg.get("hybrid_router_scale", args.hybrid_router_scale)

    args.predict_patch_size = pred_cfg.get("patch_size", args.predict_patch_size)
    if args.mode == "predict":
        args.limit = pred_cfg.get("limit", args.limit)

    for key, value in overrides.items():
        setattr(args, key, value)

    # Resolve repository-relative paths so configs remain portable across machines.
    # 配置里优先写仓库相对路径，运行时再解析成绝对路径，方便本机/A6000 迁移。
    for key in ["data_root", "test_root", "train_list", "out_dir", "zip", "ckpt", "warm_start"]:
        value = getattr(args, key)
        if value and not Path(value).is_absolute():
            setattr(args, key, str(ROOT / value))
    return args


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="", help="YAML config for reproducible runs")
    p.add_argument("--profile", action="append", default=[], help="Optional YAML profile override; can be used multiple times")
    p.add_argument("--experiment-name", default="manual")
    p.add_argument("--mode", choices=["train", "predict", "zip", "validate-zip"], default="train")
    p.add_argument("--data-root", default="dataset_train")
    p.add_argument("--test-root", default="dataset_test_noisy")
    p.add_argument("--train-list", default="starter_code/datalist/train.txt")
    p.add_argument("--out-dir", default="results/denoise_baseline")
    p.add_argument("--zip", default="result_denoise_baseline.zip")
    p.add_argument("--ckpt", default="experiments/denoise_baseline/best.pkl")
    p.add_argument("--warm-start", default="", help="Optional compatible checkpoint to initialize model before training")
    p.add_argument("--num-points", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--cache-clean", action="store_true", help="cache normalized clean OBJ vertices in memory during synthetic training")
    p.add_argument("--prefetch-workers", type=int, default=0, help="background NumPy batch workers for synthetic training")
    p.add_argument("--prefetch-queue-size", type=int, default=4)
    p.add_argument("--profile-times", action="store_true", help="log data/compute/step timing")
    p.add_argument("--profile-system-every", type=int, default=0, help="log nvidia-smi/CPU usage every N logged steps; 0 disables")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--feat-dim", type=int, default=256)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--loss-type", choices=["mse", "huber", "smooth_l1"], default="mse", help="Point-wise offset regression loss; mse preserves baseline behavior")
    p.add_argument("--huber-delta", type=float, default=0.01, help="Delta/beta for huber or smooth_l1 offset loss")
    p.add_argument("--lr-scheduler", choices=["none", "constant", "cosine", "warmup_cosine"], default="none", help="Step LR scheduler; none preserves baseline behavior")
    p.add_argument("--warmup-steps", type=int, default=0, help="Linear LR warmup steps before optional cosine decay")
    p.add_argument("--eta-min-ratio", type=float, default=0.05, help="Minimum LR ratio for cosine scheduler")
    p.add_argument("--noise-min", type=float, default=0.005)
    p.add_argument("--noise-max", type=float, default=0.02)
    p.add_argument("--cd-weight", type=float, default=0.0)
    p.add_argument("--identity-weight", type=float, default=0.0, help="Regularize predictions toward noisy input; useful as low-noise identity pressure")
    p.add_argument("--movement-weight", type=float, default=0.0, help="Regularize residual movement magnitude")
    p.add_argument("--pwsenel", action="store_true")
    p.add_argument("--staas", action="store_true", help="Enable ST-AAS geometry residual branch")
    p.add_argument("--staas-fusion", action="store_true", help="Fuse ST-AAS geometry statistics into the neural denoising head")
    p.add_argument("--staas-v2-gate", action="store_true", help="Enable ST-AAS v2 noisy-conditioned residual gate")
    p.add_argument("--staas-v2-geo-weight", type=float, default=0.25)
    p.add_argument("--staas-v2-gate-min", type=float, default=0.0)
    p.add_argument("--staas-v2-gate-max", type=float, default=1.0)
    p.add_argument("--staas-v2-noise-ref-low", type=float, default=0.010)
    p.add_argument("--staas-v2-noise-ref-high", type=float, default=0.030)
    p.add_argument("--move-gate", action="store_true", help="Enable per-point offset movement gate")
    p.add_argument("--pwsenel-v2", action="store_true", help="Enable PW-SENEL v2 explicit noise/edge confidence gate")
    p.add_argument("--pwsenel-v2-edge-lock", type=float, default=0.7)
    p.add_argument("--pwsenel-v2-gate-scale", type=float, default=0.5)
    p.add_argument("--residual-clip", type=float, default=0.0, help="Clip per-point residual L2 norm; 0 disables clipping")
    p.add_argument("--adaptive-clip", action="store_true", help="Enable cloud-scale adaptive residual clipping")
    p.add_argument("--adaptive-clip-min", type=float, default=0.006)
    p.add_argument("--adaptive-clip-max", type=float, default=0.020)
    p.add_argument("--adaptive-clip-ref-low", type=float, default=0.022)
    p.add_argument("--adaptive-clip-ref-mid", type=float, default=0.030)
    p.add_argument("--adaptive-clip-ref-high", type=float, default=0.040)
    p.add_argument("--adaptive-clip-mid", type=float, default=0.010)
    p.add_argument("--noise-aware-move-gate", action="store_true", help="Scale move_gate by per-cloud KNN noise/spacing estimate")
    p.add_argument("--noise-aware-gate-min", type=float, default=0.45)
    p.add_argument("--noise-aware-gate-ref-low", type=float, default=0.022)
    p.add_argument("--noise-aware-gate-ref-high", type=float, default=0.036)
    p.add_argument("--hybrid-safe-strong", action="store_true", help="Blend conservative PW-SENEL offset with strong move_gate branch")
    p.add_argument("--hybrid-router-scale", type=float, default=1.0)
    p.add_argument("--staas-strength", type=float, default=1.0)
    p.add_argument("--staas-tau0", type=float, default=0.02)
    p.add_argument("--staas-tau-min", type=float, default=0.005)
    p.add_argument("--staas-tau-max", type=float, default=0.08)
    p.add_argument("--predict-patch-size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--cpu", action="store_true", help="Run Jittor in CPU mode for local smoke tests")
    return p


def main() -> None:
    parser = build_parser()
    defaults = vars(parser.parse_args([]))
    args = parser.parse_args()
    # argparse 无法直接区分“用户显式传入”和“默认值”，所以这里先记录显式覆盖。
    args._cli_overrides = {
        k: v for k, v in vars(args).items()
        if k in defaults and v != defaults[k] and k != "config"
    }
    args = apply_config(args)
    set_seed(args.seed)
    if args.cpu:
        jt.flags.use_cuda = 0
    else:
        jt.flags.use_cuda = 1
    if args.mode == "train":
        train(args)
    elif args.mode == "predict":
        predict(args)
    elif args.mode == "zip":
        make_zip(args)
    else:
        validate_zip(args)


if __name__ == "__main__":
    main()
