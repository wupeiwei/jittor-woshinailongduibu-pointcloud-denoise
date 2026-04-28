# ST-AAS / PW-SENEL CVMJ Project Plan

> Working title: **Structure Tensor-guided Adaptive Anisotropic Softmax for Lightweight Point Cloud Denoising**
>
> Chinese title draft: **面向轻量点云降噪的结构张量引导自适应各向异性 Softmax 方法**

## 1. Project Goal

This project aims to turn the original **Softmax denoising + edge locking** intuition into a competition-ready and paper-ready point cloud denoising method.

Short-term goal:

- Improve the Jittor point cloud denoising competition result.
- Keep the method lightweight, reproducible, and compatible with RTX 5060 Ti / RTX A6000 environments.

Long-term goal:

- Prepare a clean algorithmic contribution that may support a CVMJ invitation paper if the competition result is strong enough.

Core positioning:

- Lightweight local geometric operator.
- Training-free or plug-in neural module.
- Single-neighborhood search design.
- Robust denoising with sharp structure preservation.

## 2. Original Motivation: Denoise-and-Lock Principle

The initial idea is a **denoise-and-lock** principle:

- Noise should be softly suppressed.
- Sharp structural cues should be protected from being smoothed away.

The early prototype expressed this as:

```text
Softmax denoising + MaxPooling edge locking
```

However, raw MaxPooling over local extrema has two major risks:

1. **Outlier locking:** high-noise outliers may become local extrema and be incorrectly preserved.
2. **Macro-collapse:** repeated local smoothing may shrink or collapse large-scale structures.

Therefore, the project upgrades raw extremum locking into:

```text
Structure Tensor-guided edge-aware smoothing suppression
```

## 3. Main Method Direction

The proposed method is tentatively named:

```text
ST-AAS: Structure Tensor-guided Adaptive Anisotropic Softmax
```

Competition/internal naming may continue to use:

```text
PW-SENEL / PW-AASENEL
```

The method should avoid heavy multi-scale search and normal dependency. The current preferred design is:

```text
Single KNN
+ density-adaptive softmax temperature
+ local structure tensor shape confidence
+ anisotropic / ellipsoidal softmax metric
+ edge-aware smoothing suppression
```

## 4. Design Principles

### 4.1 Keep

- Single KNN neighborhood.
- Relative coordinate normalization.
- Density-adaptive softmax temperature.
- Structure tensor shape analysis.
- Soft confidence instead of hard noise type classification.
- Edge-aware smoothing suppression.
- Chunked inference for large point clouds.

### 4.2 Avoid for the core method

- Repeated real multi-scale KNN / radius search.
- Raw MaxPooling over unprotected extrema.
- Strong dependency on estimated normals.
- Dynamic K with variable tensor shapes.
- Hard Gaussian / impulse / mixed noise classification.
- Custom CUDA kernels in the first implementation.
- iKD-Tree as a core competition dependency.
- Aggressive voxel downsampling that destroys fine structures.

## 5. Local Geometry Statistics

For each point `p_i`, build a fixed-size KNN neighborhood:

```text
N_i = KNN(p_i, K)
r_ij = p_j - p_i
```

Estimate local scale:

```text
scale_i = median_j ||r_ij||
```

Estimate local structure tensor / covariance:

```text
mu_i = mean_j p_j
C_i = 1/K * sum_j (p_j - mu_i)(p_j - mu_i)^T
```

Let eigenvalues be sorted as:

```text
lambda_1 >= lambda_2 >= lambda_3 >= 0
```

Use shape descriptors:

```text
linearity  = (lambda_1 - lambda_2) / (lambda_1 + eps)
planarity  = (lambda_2 - lambda_3) / (lambda_1 + eps)
scattering = lambda_3 / (lambda_1 + eps)
```

Interpretation:

- High `linearity`: edge-like / ridge-like structure.
- High `planarity`: locally planar surface.
- High `scattering`: noisy, volumetric, or unstable neighborhood.

## 6. Density-Adaptive Softmax Temperature

Avoid a globally fixed softmax temperature. Use local density / local scale to adapt temperature.

Preferred stable form:

```text
density_ratio_i = clamp(scale_avg / (scale_i + eps), r_min, r_max)
tau_i = clamp(tau_0 / sqrt(density_ratio_i), tau_min, tau_max)
```

Intuition:

- High-density area: smaller `scale_i`, larger `density_ratio_i`, smaller `tau_i`, sharper softmax.
- Low-density area: larger `tau_i`, softer aggregation, reducing sparse-region misjudgment.

## 7. Edge-aware Smoothing Suppression

Given a softmax denoised candidate:

```text
s_i = sum_j w_ij p_j
smooth_offset_i = s_i - p_i
```

Use structure confidence to suppress oversmoothing near edges:

```text
edge_conf_i = f(linearity_i, planarity_i, scattering_i)
pred_i = p_i + (1 - edge_conf_i) * smooth_offset_i
```

Important conceptual change:

```text
Edge locking does not mean pulling points toward raw extrema.
Edge locking means reducing destructive smoothing around sharp structures.
```

## 8. Anisotropic / Ellipsoidal Softmax Metric

Basic isotropic softmax:

```text
w_ij = softmax(-||r_ij||^2 / tau_i)
```

Target anisotropic form:

```text
d_ij = r_ij^T A_i r_ij
w_ij = softmax(-d_ij / tau_i)
```

Where `A_i` is induced by the local structure tensor eigenvectors/eigenvalues.

Intuition:

- Along surface or edge direction: allow stronger aggregation.
- Across structure boundary: suppress aggregation.
- In noisy scattering regions: increase robust penalty.

## 9. Perception-Decision-Execution-Stability Loop

The current explanation framework is:

```text
Perception -> Decision -> Execution -> Stability feedback
```

### 9.1 Perception layer

```text
Structure Tensor
```

The operator first observes local point distribution instead of judging one point in isolation. Eigenvalue descriptors provide soft geometric states:

- `linearity`: edge-like or ridge-like tendency.
- `planarity`: surface-like tendency.
- `scattering`: noisy / volumetric / unstable tendency.

### 9.2 Decision layer

```text
Density-adaptive softmax temperature + edge-aware smoothing suppression
```

Density-adaptive temperature answers:

```text
How sharp should the local softmax be here?
```

Structure-aware edge confidence answers:

```text
How much smoothing is safe here?
```

This is the key upgrade from the original plain Softmax + MaxPool idea: the method no longer blindly filters and blindly locks extrema. It first estimates local geometry, then modulates denoising strength.

### 9.3 Execution layer

```text
Adaptive softmax local aggregation
```

The execution layer produces a denoised candidate from one fixed KNN neighborhood. In v0 this is still isotropic; in v1 it becomes anisotropic / ellipsoidal.

### 9.4 Stability layer

```text
Edge-aware residual suppression
```

The final update is residual and conservative:

```text
pred_i = p_i + (1 - edge_conf_i) * (smooth_i - p_i)
```

This prevents the method from becoming a raw local filter. Around edge-like structures, it suppresses destructive over-smoothing instead of pulling points toward noisy extrema.

## 10. Implementation Stages

### Stage 0: Baseline stability

- Keep current denoise baseline reproducible.
- Continue 5060 Ti / A6000 environment validation.
- Collect first complete training logs.

### Stage 1: ST-AAS v0

Minimal module:

- Fixed KNN from existing feature extractor.
- Local density / scale estimate.
- Density-adaptive `tau_i`.
- Structure tensor eigenvalue descriptors.
- `edge_conf` smoothing suppression.

No full anisotropic matrix yet.

### Stage 2: ST-AAS v1

Add:

- Ellipsoidal softmax metric.
- Tensor-guided anisotropic distance.
- More complete ablation.

### Stage 3: Paper version

Add:

- Training-free variant if feasible.
- Plug-in neural variant.
- Runtime / memory benchmark.
- Large-scale chunked inference evaluation.
- Strong visualizations and ablation tables.

## 11. Ablation Plan

Minimum ablation list:

```text
1. Baseline denoiser
2. Baseline + original PW-SENEL / MaxPool version
3. Baseline + density-adaptive tau only
4. Baseline + structure tensor edge_conf only
5. Baseline + ST-AAS v0
6. Baseline + ST-AAS v1 anisotropic metric
```

Metrics:

- Chamfer Distance (CD)
- Point-to-Surface Distance (P2S)
- Relative improvement over noisy input
- Inference time
- Peak VRAM
- Edge preservation visual comparison

## 12. Paper Storyline

Problem conflict:

```text
Denoising vs sharp feature preservation
Local robustness vs global stability
Model accuracy vs computational cost
```

Existing method limitations:

- Traditional geometry methods are lightweight but often oversmooth sharp features.
- Heavy deep denoisers may work well but require substantial training and compute.
- Raw local extrema / MaxPooling can be misled by outliers.
- Normal-based constraints can fail under high noise due to pseudo-normal estimation.

Proposed claim:

```text
We propose a lightweight structure-aware local operator that performs adaptive anisotropic softmax denoising within a single KNN neighborhood, preserving sharp structures without repeated multi-scale search or strong normal dependency.
```

## 13. Current Risks

- Structure tensor may still be unstable under extreme noise.
- Eigenvalue computation cost must be checked in Jittor.
- Anisotropic matrix version may introduce numerical instability.
- Edge suppression may under-denoise real noisy edge regions.
- Competition score may not align perfectly with visual edge preservation.

## 14. Immediate Next Actions

1. Keep collecting 5060 Ti baseline training result.
2. Implement ST-AAS v0 as a switchable module, not replacing the baseline.
3. Add config file for ST-AAS experiments.
4. Run small smoke tests locally.
5. Run ablation on RTX 5060 Ti.
6. If promising, scale to A6000.
