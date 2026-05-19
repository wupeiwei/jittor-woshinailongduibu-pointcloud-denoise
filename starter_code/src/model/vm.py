import os
from math import ceil
from typing import Dict, List

import jittor as jt
import numpy as np

from .feature import FeatureExtraction, Decoder
from .spec import ModelSpec

from ..data.asset import Asset

def get_random_indices(n, m):
    assert m < n
    idx = np.random.permutation(n)[:m]
    return jt.array(idx).int32()

class VelocityModule(ModelSpec):
    
    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        
        cfg = self.model_config
        # geometry
        self.frame_knn = cfg['frame_knn']
        self.num_train_points = cfg['num_train_points']
        
        # score-matching
        self.dsm_sigma = cfg['dsm_sigma']

        # networks
        self.encoder = FeatureExtraction(
            k=self.frame_knn,
            input_dim=3,
            embedding_dim=cfg['feat_embedding_dim']
        )
        
        self.decoder = Decoder(
            z_dim=self.encoder.embedding_dim,
            dim=3,
            out_dim=3,
            hidden_size=cfg['decoder_hidden_dim'],
        )
    
    def get_supervised_loss(self, pc_noisy, pc_mix, pc_clean):
        """
        pcl_noisy: (B, N, 3)
        pcl_clean: (B, N, 3)
        """
        B, N_noisy, d = pc_mix.shape
        
        pnt_idx = get_random_indices(N_noisy, self.num_train_points)
        
        # Feature extraction
        feat = self.encoder(pc_mix)  # (B, N, F)
        F_dim = feat.shape[2]
        
        # gather
        feat = feat[:, pnt_idx, :]
        pc_noisy = pc_noisy[:, pnt_idx, :]
        pc_mix = pc_mix[:, pnt_idx, :]
        pc_clean = pc_clean[:, pnt_idx, :]
        
        # target
        grad_dir_t_target = pc_clean - pc_noisy
        
        # decoder
        pred_dir = self.decoder(
            c=feat.reshape(-1, F_dim)
        ).reshape(B, len(pnt_idx), d) # type: ignore
        
        loss = (((pred_dir - grad_dir_t_target) ** 2.0) / self.dsm_sigma).sum(dim=-1).mean()
        
        return loss

    def denoise_langevin_dynamics(self, pcl_noisy, num_steps: int=4):
        """
        pcl_noisy: (B, N, 3)
        """
        B, N, d = pcl_noisy.shape
        with jt.no_grad():
            pcl_next = pcl_noisy.clone()
            for it in range(num_steps):
                feat = self.encoder(pcl_next)  # (B, N, F)
                F_dim = feat.shape[2]
                
                pred_dir = self.decoder(
                    c=feat.reshape(-1, F_dim)
                ).reshape(B, N, d)
                
                pcl_next = pcl_next + (1.0 / num_steps) * pred_dir
        return pcl_next, None
    
    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch['pc_noisy'].shape[-2]
        pc_noisy = batch['pc_noisy'].reshape(-1, patch_size, 3)
        pc_mix = batch['pc_mix'].reshape(-1, patch_size, 3)
        pc_clean = batch['pc_clean'].reshape(-1, patch_size, 3)
        loss = self.get_supervised_loss(
            pc_noisy=pc_noisy,
            pc_mix=pc_mix,
            pc_clean=pc_clean,
        )
        return {"loss": loss}
    
    def execute(self, **kwargs) -> Dict: # type: ignore
        return self.training_step(**kwargs)
    
    @jt.no_grad()
    def predict_step(self, batch: Dict) -> List[Dict]:
        pc_noisy_batch = batch['pc_noisy']
        assert pc_noisy_batch.ndim == 3

        stream_mode = os.environ.get("VM_STITCHING", "auto").lower()
        streaming_min_points = int(os.environ.get("VM_STREAMING_MIN_POINTS", "32768"))

        num_steps = 1
        res = []
        for i, pc_noisy in enumerate(pc_noisy_batch):
            pc_next = pc_noisy
            for it in range(num_steps):
                # VM_STITCHING 控制“拼接实现”而不是模型权重：
                # - dense fixed-stitch：适合小云，逻辑最接近官方原实现；
                # - streaming fixed-stitch：适合大云，避免构造 P x N 稠密矩阵。
                use_streaming = (
                    stream_mode == "streaming"
                    or (stream_mode == "auto" and pc_next.shape[0] >= streaming_min_points)
                )
                denoise_fn = patch_based_denoise_streaming if use_streaming else patch_based_denoise
                pc_next = denoise_fn(
                    model=self,
                    pcl_noisy=pc_next,
                    patch_size=1000,
                    seed_k=6,
                    seed_k_alpha=1,
                )
            pc_denoised = pc_next.detach().numpy()
            res.append({"pc_denoised": pc_denoised})
        return res
    
    def process_fn(self, batch: List[Asset]) -> List[Dict]:
        res = []
        for b in batch:
            if not self.is_predict():
                assert b.meta is not None
                res.append({
                    "pc_noisy": b.meta['pc_noisy'], # (num_patches, patch_size, 3)
                    "pc_clean": b.meta['pc_clean'],
                    "pc_mix": b.meta['pc_mix'],
                })
            else:
                d = {
                    "pc_noisy": b.sampled_vertices_noisy, # (N, 3)
                }
                if b.sampled_vertices is not None:
                    d["pc_clean"] = b.sampled_vertices
                res.append(d)
        return res

def farthest_point_sampling(pcls, num_pnts):
    """
    pcls: (B, N, 3)
    return:
        sampled: (B, num_pnts, 3)
        indices: (B, num_pnts)
    """
    B, N, _ = pcls.shape
    sampled = []
    indices = []
    for b in range(B):
        # 朴素 FPS：官方风格实现，点数较大时慢但确定性足够用于当前 baseline。
        pts = pcls[b]  # (N, 3)
        selected = []
        dist = jt.ones((N,)) * 1e10
        farthest = 0
        for i in range(num_pnts):
            selected.append(farthest)
            centroid = pts[farthest]  # (3,)
            d = ((pts - centroid) ** 2).sum(dim=1)
            dist = jt.minimum(dist, d)
            farthest, _ = jt.argmax(dist, dim=-1)
            farthest = farthest.item()
        idx = jt.array(selected).int32()
        sampled.append(pts[idx][None, ...])
        indices.append(idx[None, ...])
    sampled = jt.concat(sampled, dim=0)
    indices = jt.concat(indices, dim=0)
    return sampled, indices

def knn_points(x, y, k):
    """
    x: (B, P, 3)
    y: (B, N, 3)
    return:
        dist: (B, P, k)
        idx:  (B, P, k)
        nn:   (B, P, k, 3)
    """
    # 这里直接构造全量 pairwise distance，适合 patch seed 数较小的官方 VM 推理。
    dist = ((x.unsqueeze(2) - y.unsqueeze(1)) ** 2).sum(-1)
    dist_k, idx = jt.topk(dist, k=k, dim=-1, largest=False)
    B = x.shape[0]
    nn = []
    for b in range(B):
        nn.append(y[b][idx[b]])
    nn = jt.stack(nn, dim=0)
    return dist_k, idx, nn

def patch_based_denoise(model: VelocityModule, pcl_noisy, patch_size=1000, seed_k=6, seed_k_alpha=1) -> jt.Var:
    """
    pcl_noisy: (N, 3)

    Official baseline note:
    The original patch stitching assumes every input point appears in at
    least one FPS/KNN patch. In practice this can be false for large clouds:
    uncovered points get an arbitrary best patch id, the membership mask is
    empty, and the final concat silently drops those points.  We repair the
    coverage before denoising by adding KNN patches centered at uncovered
    points.  This keeps the model/checkpoint unchanged while guaranteeing a
    one-to-one output point for every input point.
    """
    assert len(pcl_noisy.shape) == 2
    
    N, d = pcl_noisy.shape
    num_patches = int(seed_k * N / patch_size)
    pcl_noisy = pcl_noisy.unsqueeze(0)  # (1, N, 3)
    
    seed_pnts, seed_idx = farthest_point_sampling(pcl_noisy, num_patches)
    patch_dists, point_idxs, patches = knn_points(seed_pnts, pcl_noisy, patch_size)

    # Coverage repair: add one seed-centered patch for each point that was
    # not selected by any initial patch. This fixes the dropped-point bug at
    # the source instead of padding the final file after prediction.
    # 中文说明：先检查每个原始点是否至少出现在一个 KNN patch 里。
    # 若没有覆盖，就以这些漏点自身作为额外 seed 再取 KNN patch，从源头保证输出点数不丢失。
    covered = np.zeros((N,), dtype=np.bool_)
    covered[point_idxs.detach().numpy().reshape(-1)] = True
    missing = np.nonzero(~covered)[0]
    if len(missing) > 0:
        missing_idx = jt.array(missing).int32().reshape(1, -1)
        missing_seed_pnts = pcl_noisy[0][missing_idx[0]].unsqueeze(0)
        missing_dists, missing_point_idxs, missing_patches = knn_points(
            missing_seed_pnts, pcl_noisy, patch_size
        )
        seed_pnts = jt.concat([seed_pnts, missing_seed_pnts], dim=1)
        patch_dists = jt.concat([patch_dists, missing_dists], dim=1)
        point_idxs = jt.concat([point_idxs, missing_point_idxs], dim=1)
        patches = jt.concat([patches, missing_patches], dim=1)
        num_patches = seed_pnts.shape[1]
    
    patches = patches[0]              # (P, M, 3)
    patch_dists = patch_dists[0]      # (P, M)
    point_idxs = point_idxs[0]        # (P, M)
    
    seed_expand = seed_pnts.squeeze().unsqueeze(1).broadcast(patches.shape)
    patches = patches - seed_expand
    
    patch_dists = patch_dists / (patch_dists[:, -1:].broadcast(patch_dists.shape) + 1e-8)
    
    all_dists = jt.ones((num_patches, N)) * 1e10
    
    for i in range(num_patches):
        all_dists[i][point_idxs[i]] = patch_dists[i]
        
    # 对每个原始点选择归一化 patch 距离最小的 patch 作为最终拼接来源。
    # exp(-dist) 与取最小 dist 等价；保留该写法是为了尽量贴近原 dense 逻辑。
    weights = jt.exp(-all_dists)
    best_weights_idx, _ = jt.argmax(weights, dim=0)
    patches_denoised = []
    
    i = 0
    patch_step = int(ceil(N / (seed_k_alpha * patch_size)))
    assert patch_step > 0
    while i < num_patches:
        curr = patches[i:i+patch_step]
        try:
            out, _ = model.denoise_langevin_dynamics(curr)
        except Exception as e:
            print("Denoise error:", e)
            return None
        patches_denoised.append(out)
        i += patch_step
    
    patches_denoised = jt.concat(patches_denoised, dim=0)
    patches_denoised = patches_denoised + seed_expand
    pcl_out = []
    for pidx in range(N):
        # dense fixed-stitch 逐点取回所属 patch 内的 denoised 坐标，保持输入/输出一一对应。
        patch_id = best_weights_idx[pidx].item()
        mask = (point_idxs[patch_id] == pidx)
        stitched = patches_denoised[patch_id][mask]
        if stitched.shape[0] == 0:
            # Defensive fallback: coverage repair should make this unreachable,
            # but never drop points from a submission artifact.
            # 理论上覆盖修复后不会进入；保留兜底是为了提交包绝不静默少点。
            stitched = pcl_noisy[0][pidx:pidx+1]
        elif stitched.shape[0] > 1:
            stitched = stitched[:1]
        pcl_out.append(stitched)
    pcl_out = jt.concat(pcl_out, dim=0)
    assert pcl_out.shape[0] == N, f"patch stitching changed point count: got {pcl_out.shape[0]}, expected {N}"
    return pcl_out


def _streaming_best_assignment(point_idxs_np: np.ndarray, patch_dists_np: np.ndarray, n: int):
    """Find each input point's nearest normalized patch assignment without P x N storage.

    This is equivalent to the dense stitching assignment in `patch_based_denoise`:

        all_dists = ones((P, N)) * 1e10
        all_dists[i][point_idxs[i]] = patch_dists[i]
        best = argmax(exp(-all_dists), dim=0)

    but keeps only O(N) arrays: best distance, best patch id and local index.
    """
    # streaming 版本不分配 all_dists(P, N)，只维护每个原始点当前最优 patch。
    # 返回 best_patch/best_local 后，后续可以边 denoise patch chunk 边回填输出数组。
    best_dist = np.full(n, np.inf, dtype=np.float32)
    best_patch = np.full(n, -1, dtype=np.int32)
    best_local = np.full(n, -1, dtype=np.int32)
    for i in range(point_idxs_np.shape[0]):
        idx = point_idxs_np[i]
        dist = patch_dists_np[i]
        better = dist < best_dist[idx]
        if np.any(better):
            pts = idx[better]
            best_dist[pts] = dist[better]
            best_patch[pts] = i
            best_local[pts] = np.nonzero(better)[0].astype(np.int32)
    return best_dist, best_patch, best_local


def patch_based_denoise_streaming(model: VelocityModule, pcl_noisy, patch_size=1000, seed_k=6, seed_k_alpha=1) -> jt.Var:
    """Patch denoise with streaming fixed-stitch assignment for large clouds.

    This function is intentionally additive: `patch_based_denoise` remains the
    default dense implementation.  The model, checkpoint, FPS seeds, KNN
    patches, normalization and Langevin denoising are unchanged.  Only the
    stitching assignment is changed from dense `all_dists(P, N)` storage to O(N)
    streaming arrays, which avoids the 50w-point memory cliff observed in large
    cloud pressure tests.
    """
    assert len(pcl_noisy.shape) == 2

    N, d = pcl_noisy.shape
    num_patches = int(seed_k * N / patch_size)
    pcl_noisy = pcl_noisy.unsqueeze(0)  # (1, N, 3)

    seed_pnts, seed_idx = farthest_point_sampling(pcl_noisy, num_patches)
    patch_dists, point_idxs, patches = knn_points(seed_pnts, pcl_noisy, patch_size)

    # Keep the same coverage repair as the dense fixed-stitch path: guarantee
    # every input point has at least one candidate patch before assignment.
    # 与 dense 路径保持完全相同的覆盖修复，避免两个 stitching 分支输出数量语义不同。
    covered = np.zeros((N,), dtype=np.bool_)
    covered[point_idxs.detach().numpy().reshape(-1)] = True
    missing = np.nonzero(~covered)[0]
    if len(missing) > 0:
        missing_idx = jt.array(missing).int32().reshape(1, -1)
        missing_seed_pnts = pcl_noisy[0][missing_idx[0]].unsqueeze(0)
        missing_dists, missing_point_idxs, missing_patches = knn_points(
            missing_seed_pnts, pcl_noisy, patch_size
        )
        seed_pnts = jt.concat([seed_pnts, missing_seed_pnts], dim=1)
        patch_dists = jt.concat([patch_dists, missing_dists], dim=1)
        point_idxs = jt.concat([point_idxs, missing_point_idxs], dim=1)
        patches = jt.concat([patches, missing_patches], dim=1)
        num_patches = seed_pnts.shape[1]

    patches = patches[0]              # (P, M, 3)
    patch_dists = patch_dists[0]      # (P, M)
    point_idxs = point_idxs[0]        # (P, M)

    seed_expand = seed_pnts.squeeze().unsqueeze(1).broadcast(patches.shape)
    patches = patches - seed_expand
    patch_dists = patch_dists / (patch_dists[:, -1:].broadcast(patch_dists.shape) + 1e-8)

    point_idxs_np = point_idxs.detach().numpy().astype(np.int64, copy=False)
    patch_dists_np = patch_dists.detach().numpy().astype(np.float32, copy=False)
    # CPU/NumPy 上计算最优归属，换取显存稳定；模型前向仍由 Jittor 执行。
    _, best_patch, best_local = _streaming_best_assignment(point_idxs_np, patch_dists_np, N)
    if np.any(best_patch < 0):
        raise RuntimeError("streaming patch stitching left uncovered points")

    patch_step = int(ceil(N / (seed_k_alpha * patch_size)))
    assert patch_step > 0
    pcl_out_np = np.empty((N, d), dtype=np.float32)
    filled = np.zeros(N, dtype=np.bool_)

    i = 0
    while i < num_patches:
        end = min(i + patch_step, num_patches)
        curr = patches[i:end]
        try:
            out, _ = model.denoise_langevin_dynamics(curr)
        except Exception as e:
            print("Denoise error:", e)
            return None
        out = out + seed_expand[i:end]
        out_np = out.numpy().astype(np.float32, copy=False)

        # 当前 chunk 只回填归属到 [i, end) patch 的原始点；其余点等对应 chunk 再填。
        pts = np.nonzero((best_patch >= i) & (best_patch < end))[0]
        if len(pts) > 0:
            local_patch = best_patch[pts] - i
            local_idx = best_local[pts]
            pcl_out_np[pts] = out_np[local_patch, local_idx]
            filled[pts] = True
        i = end

    if not filled.all():
        raise RuntimeError(f"streaming patch stitching left {(~filled).sum()} output points unfilled")
    assert pcl_out_np.shape[0] == N, f"patch stitching changed point count: got {pcl_out_np.shape[0]}, expected {N}"
    return jt.array(pcl_out_np)
