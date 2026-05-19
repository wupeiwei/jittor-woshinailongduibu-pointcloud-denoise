#!/usr/bin/env python3
"""Jittor operator compatibility prototype for a StraightPCF-style move-scale gate.

This is intentionally standalone: it validates shapes, KNN gathering, bounded
scale prediction, offset composition, loss, and backward/optimizer step before
we wire the idea into the main denoiser.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jittor as jt
from jittor import nn

from denoise_baseline import FeatureExtraction, get_knn_idx, set_seed


class StraightMoveScaleProto(nn.Module):
    """Minimal direction + bounded distance/move-scale prototype.

    direction: per-point 3D residual direction.
    scale: per-point bounded scalar in [scale_min, scale_max].
    cloud_scale: optional low-noise-aware multiplier from KNN spacing.
    """

    def __init__(
        self,
        k: int = 8,
        feat_dim: int = 64,
        hidden: int = 64,
        scale_min: float = 0.0,
        scale_max: float = 1.0,
        low_ref: float = 0.010,
        high_ref: float = 0.030,
    ):
        super().__init__()
        self.k = k
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.low_ref = low_ref
        self.high_ref = high_ref
        self.encoder = FeatureExtraction(k=k, input_dim=3, embedding_dim=feat_dim)
        self.direction_head = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )
        self.scale_head = nn.Sequential(
            nn.Linear(feat_dim + 1, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )

    def gather_neighbors(self, x: jt.Var, idx: jt.Var) -> jt.Var:
        B, N, C = x.shape
        base = (jt.arange(B) * N).reshape(B, 1, 1)
        flat_idx = (idx + base).reshape(-1)
        flat = x.reshape(B * N, C)
        return flat[flat_idx].reshape(B, N, idx.shape[-1], C)

    def execute(self, noisy: jt.Var, return_stats: bool = False):
        feat = self.encoder(noisy)
        B, N, C = feat.shape
        idx = get_knn_idx(noisy, noisy, self.k + 1)[:, :, 1:]
        neigh = self.gather_neighbors(noisy, idx)
        local_dist = (((neigh - noisy.unsqueeze(2)) ** 2).sum(dim=-1, keepdims=True) + 1e-12) ** 0.5
        mean_dist = local_dist.mean(dim=2)  # (B,N,1)
        cloud_dist = mean_dist.mean(dim=1, keepdims=True)  # (B,1,1)
        cloud_t = jt.clamp((cloud_dist - self.low_ref) / (self.high_ref - self.low_ref + 1e-12), 0.0, 1.0)

        direction = self.direction_head(feat.reshape(B * N, C)).reshape(B, N, 3)
        scale_input = jt.concat([feat, mean_dist], dim=-1)
        raw_scale = self.scale_head(scale_input.reshape(B * N, C + 1)).reshape(B, N, 1)
        bounded_scale = self.scale_min + (self.scale_max - self.scale_min) * raw_scale
        # StraightPCF-inspired guard: low-noise clouds should naturally move less.
        move_scale = bounded_scale * cloud_t
        offset = direction * move_scale
        pred = noisy + offset
        if return_stats:
            return pred, {
                "direction": direction,
                "raw_scale": raw_scale,
                "move_scale": move_scale,
                "mean_dist": mean_dist,
                "cloud_t": cloud_t,
                "cloud_dist": cloud_dist,
                "offset": offset,
            }
        return pred


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--points", type=int, default=128)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--feat-dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260512)
    ap.add_argument("--mode", choices=["random", "sphere"], default="sphere")
    ap.add_argument("--sigmas", default="0.006,0.030")
    ap.add_argument("--low-ref", type=float, default=0.010)
    ap.add_argument("--high-ref", type=float, default=0.030)
    args = ap.parse_args()

    set_seed(args.seed)
    jt.flags.use_cuda = 1

    if args.mode == "sphere":
        # Structured point cloud, closer to normalized ShapeNet surface spacing
        # than uniform random volume points.
        xyz = jt.randn((args.batch, args.points, 3))
        clean = xyz / ((((xyz ** 2).sum(dim=-1, keepdims=True)) + 1e-12) ** 0.5)
    else:
        clean = jt.rand((args.batch, args.points, 3)) * 2.0 - 1.0
    sigmas = [float(x) for x in args.sigmas.split(",") if x.strip()]
    if len(sigmas) == args.batch:
        sigma = jt.array(sigmas).reshape(args.batch, 1, 1)
    else:
        sigma = jt.ones((args.batch, 1, 1)) * sigmas[0]
    noisy = clean + jt.randn(clean.shape) * sigma

    model = StraightMoveScaleProto(
        k=args.k,
        feat_dim=args.feat_dim,
        hidden=args.hidden,
        low_ref=args.low_ref,
        high_ref=args.high_ref,
    )
    opt = nn.Adam(model.parameters(), lr=1e-3)
    pred, stats = model(noisy, return_stats=True)
    loss = ((pred - clean) ** 2).mean()
    opt.step(loss)

    # Materialize after backward to catch both forward and optimizer issues.
    print("OK prototype_move_scale_gate")
    print("pred_shape", pred.shape)
    print("offset_shape", stats["offset"].shape)
    print("move_scale_shape", stats["move_scale"].shape)
    print("cloud_t_shape", stats["cloud_t"].shape)
    print("loss", float(loss.item()))
    print("move_scale_minmax", float(stats["move_scale"].min().item()), float(stats["move_scale"].max().item()))
    print("cloud_t", stats["cloud_t"].reshape(-1).numpy().tolist())
    print("cloud_dist", stats["cloud_dist"].reshape(-1).numpy().tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
