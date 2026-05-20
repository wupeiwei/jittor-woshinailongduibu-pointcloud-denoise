# API 参考文档

## 概述

本文档面向使用或扩展 `jittor-pointcloud-denoise` 项目的开发者，集中说明
代码库中**对外可复用**的核心模块、类和函数：

- 邻域聚合工具（`denoise_utils.gather_neighbors`）
- 数据集类（`ObjDenoiseDataset`）
- 三种几何/特征研究算子（`PWSENEL`、`PWSENELv2Gate`、`STAASv0`）
- 残差去噪主模型（`ResidualDenoiser`），含训练/推理入口函数
- 配置文件中可用的关键参数，以及 `configs/profiles/` 的用途

文档内容严格基于 `denoise_utils.py`、`denoise_baseline.py` 与
`configs/` 中的实际代码与 YAML 字段，未在此处引入未实现的接口。

> 下文所有形状记号约定：`B` 为 batch、`N` 为点数、`C` 为特征通道、
> `K` 为 KNN 邻居数。

---

## denoise_utils 模块

`denoise_utils.py` 是一个轻量纯函数模块，目前仅暴露一个跨多模型共享的
KNN 邻居聚合工具。

### `gather_neighbors(x, idx)`

按 KNN 索引，沿第二个轴聚合每个点的邻居特征。

**签名**

```python
def gather_neighbors(x: jt.Var, idx: jt.Var) -> jt.Var
```

**参数**

| 参数 | 类型 | 形状 | 说明 |
| ---- | ---- | ---- | ---- |
| `x` | `jt.Var` | `(B, N, C)` | 每点特征张量 |
| `idx` | `jt.Var` | `(B, N, K)` | KNN 索引；每个 `idx[b, n, k]` 必须落在 `[0, N)` 区间，并指向 `x[b]` 中的某一行 |

**返回值**

| 类型 | 形状 | 含义 |
| ---- | ---- | ---- |
| `jt.Var` | `(B, N, K, C)` | 满足 `out[b, n, k] = x[b, idx[b, n, k]]` 的邻居特征张量 |

**示例**

```python
import jittor as jt
from denoise_utils import gather_neighbors

# 假设特征已经计算好，并通过 KNN 得到了 idx
B, N, C, K = 2, 1024, 64, 16
x = jt.randn(B, N, C)
idx = jt.randint(0, N, shape=(B, N, K))

neigh = gather_neighbors(x, idx)
assert neigh.shape == [B, N, K, C]
```

> 在 `denoise_baseline.py` 中，`PWSENEL`、`PWSENELv2Gate`、`STAASv0` 与
> `ResidualDenoiser` 都各自保留了一个名为 `gather_neighbors` 或
> `_gather_neighbors` 的薄包装器，本质都直接调用本函数，仅为兼容旧调用点。

---

## denoise_baseline 模块

`denoise_baseline.py` 是项目主脚本，既包含模型定义，也包含
`train` / `predict` / `zip` / `validate-zip` 等命令行入口。下文按使用频率
列出对外比较稳定的类与函数。

### `ObjDenoiseDataset` 类

**功能**：面向 ShapeNet OBJ 干净网格的小型数据集。每个样本会从模型中按需
采样固定数量的点云，并在线合成高斯噪声，以便快速做 baseline 训练，
不依赖外部预处理流水线。

**初始化参数**

| 参数 | 类型 | 默认值 | 说明 |
| ---- | ---- | ---- | ---- |
| `data_root` | `str` | 必填 | OBJ 数据根目录；样本路径形如 `<data_root>/shapenet/<synset>/<model>/models/model_normalized.obj` |
| `list_file` | `str` | 必填 | 训练列表文件，每行一个相对路径，可以是 `shapenet/...` 或 `<synset>/<model>` |
| `num_points` | `int` | `2048` | 每次 `__getitem__` 返回的点数 |
| `noise_min` | `float` | `0.005` | 高斯噪声 σ 的下界 |
| `noise_max` | `float` | `0.02` | 高斯噪声 σ 的上界（每次随机均匀采样一个 σ） |
| `limit` | `int` | `0` | 仅取列表的前 `limit` 个样本，`0` 表示不限 |
| `cache_clean` | `bool` | `False` | 是否在内存中缓存归一化后的 clean 顶点；缓存只作用于干净网格，每次仍会重新采样并加噪 |

**关键方法**

| 方法 | 说明 |
| ---- | ---- |
| `__len__()` | 返回有效 OBJ 样本数 |
| `__getitem__(idx)` | 返回 `(noisy, clean)` 元组，二者均为形状 `(num_points, 3)` 的 `np.float32` 数组 |

**示例**

```python
from denoise_baseline import ObjDenoiseDataset

ds = ObjDenoiseDataset(
    data_root="dataset_train",
    list_file="starter_code/datalist/train.txt",
    num_points=2048,
    noise_min=0.005,
    noise_max=0.02,
    cache_clean=True,
)
noisy, clean = ds[0]
```

---

### `PWSENEL` 类

**功能**：PW-SENEL（Softmax Noise Elimination + MaxPool Edge Locking）
特征精炼算子。给定每点特征与坐标后：

- Softmax 分支学习邻居置信度，抑制可疑噪声响应；
- MaxPool 分支保留高响应的局部边缘/几何线索；
- 二者融合后**残差加回**到输入特征上，便于做无害消融。

继承 `jittor.nn.Module`，前向方法名为 `execute`。

**初始化参数**

| 参数 | 类型 | 默认值 | 说明 |
| ---- | ---- | ---- | ---- |
| `channels` | `int` | 必填 | 输入/输出特征通道 `C` |
| `k` | `int` | `16` | KNN 邻居数（不含中心点本身） |

**`execute(feat, points)` 输入输出**

| 名称 | 类型 | 形状 | 含义 |
| ---- | ---- | ---- | ---- |
| 输入 `feat` | `jt.Var` | `(B, N, C)` | 每点特征 |
| 输入 `points` | `jt.Var` | `(B, N, 3)` | 每点坐标，用于 KNN 与几何编码 |
| 返回 | `jt.Var` | `(B, N, C)` | 残差精炼后的新特征，等于 `feat + fused` |

---

### `PWSENELv2Gate` 类

**功能**：PW-SENEL v2 显式置信度门控。**它不直接重写偏移**，而是输出一个
`move_gate`，与神经偏移头相乘，从而：

- `noise_conf` 学习哪些点更需要去噪；
- `edge_conf` 用 MaxPool 风格的局部几何响应锁定锐利结构；
- 最终 `move_gate = gate_scale * noise_conf * (1 - edge_lock_strength * edge_conf)`，
  并被 `clamp` 到 `[0, 1]`。

继承 `jittor.nn.Module`，前向方法名为 `execute`。

**初始化参数**

| 参数 | 类型 | 默认值 | 说明 |
| ---- | ---- | ---- | ---- |
| `channels` | `int` | 必填 | 输入特征通道 `C` |
| `k` | `int` | `16` | KNN 邻居数 |
| `edge_lock_strength` | `float` | `0.7` | 边缘锁定强度，越大越压制边缘附近的偏移 |
| `gate_scale` | `float` | `0.5` | 门控全局缩放因子，控制偏移最大可放大幅度 |

**`execute(feat, points, return_stats=False)` 输入输出**

| 名称 | 类型 | 形状 | 含义 |
| ---- | ---- | ---- | ---- |
| 输入 `feat` | `jt.Var` | `(B, N, C)` | 每点特征 |
| 输入 `points` | `jt.Var` | `(B, N, 3)` | 每点坐标 |
| 输入 `return_stats` | `bool` | 标量 | 是否同时返回中间统计字典 |
| 返回 `move_gate` | `jt.Var` | `(B, N, 1)` | 偏移门控 |
| 返回 `stats`（可选） | `dict` | — | 包含 `noise_conf`、`edge_conf`、`move_gate` 等张量，便于日志/可视化 |

---

### `STAASv0` 类

**功能**：ST-AAS v0（Structure Tensor-guided Adaptive Softmax）几何算子。
是一个**最小可切换**的可微/无参数几何分支，专门用于消融：

- 单层 KNN 邻域；
- 基于局部尺度/密度的自适应 softmax 温度 `tau`；
- 协方差不变量（线性度、平面度、散射度）替代特征值分解，避免依赖
  `cusolver`，在算力卡上更稳定；
- 边缘感知地抑制平滑分量，得到 `pred = points + (1 - edge_conf) * smooth_offset`。

继承 `jittor.nn.Module`，前向方法名为 `execute`。

**初始化参数**

| 参数 | 类型 | 默认值 | 说明 |
| ---- | ---- | ---- | ---- |
| `k` | `int` | `16` | KNN 邻居数 |
| `tau0` | `float` | `0.02` | 基础 softmax 温度 |
| `tau_min` | `float` | `0.005` | 自适应温度下界 |
| `tau_max` | `float` | `0.08` | 自适应温度上界 |
| `density_min` | `float` | `0.5` | 局部密度比下界（用于温度自适应） |
| `density_max` | `float` | `2.0` | 局部密度比上界 |
| `eps` | `float` | `1e-8` | 数值稳定常数 |

**`execute(points, return_stats=False)` 输入输出**

| 名称 | 类型 | 形状 | 含义 |
| ---- | ---- | ---- | ---- |
| 输入 `points` | `jt.Var` | `(B, N, 3)` | 输入点云坐标 |
| 输入 `return_stats` | `bool` | 标量 | 是否返回中间几何统计字典 |
| 返回 `pred` | `jt.Var` | `(B, N, 3)` | 经几何分支调整后的点云预测 |
| 返回 `stats`（可选） | `dict` | — | 包含 `scale`、`tau`、`linearity`、`planarity`、`scattering`、`edge_conf`、`smooth_offset` |

---

### `ResidualDenoiser` 类

**功能**：项目主模型。主路径为 `noisy -> FeatureExtraction -> 残差偏移 ->
（可选）几何/门控分支 -> pred = noisy + offset`。所有研究模块（PW-SENEL、
PW-SENEL v2、ST-AAS v0/v2、move_gate、自适应裁剪、混合 safe/strong 路由
等）都以**开关参数**形式挂接，方便做消融或回退。

主干编码器复用 `starter_code` 提供的官方 `FeatureExtraction`
（基于 EdgeConv），头部为多层 MLP 输出 3D 偏移。

继承 `jittor.nn.Module`，前向方法名为 `execute`，可作为可调用对象使用。

**初始化参数（按功能分组）**

主干结构：

| 参数 | 类型 | 默认值 | 说明 |
| ---- | ---- | ---- | ---- |
| `k` | `int` | `16` | 主干 KNN 邻居数 |
| `feat_dim` | `int` | `256` | 编码器输出特征维度 |
| `hidden` | `int` | `256` | 头部 MLP 隐藏维度 |

PW-SENEL 系列：

| 参数 | 类型 | 默认值 | 说明 |
| ---- | ---- | ---- | ---- |
| `use_pwsenel` | `bool` | `False` | 在编码器后插入 `PWSENEL` 特征精炼 |
| `use_pwsenel_v2` | `bool` | `False` | 启用 `PWSENELv2Gate` 偏移门控 |
| `pwsenel_v2_edge_lock` | `float` | `0.7` | 见 `PWSENELv2Gate.edge_lock_strength` |
| `pwsenel_v2_gate_scale` | `float` | `0.5` | 见 `PWSENELv2Gate.gate_scale` |

ST-AAS 系列：

| 参数 | 类型 | 默认值 | 说明 |
| ---- | ---- | ---- | ---- |
| `use_staas` | `bool` | `False` | 启用 `STAASv0` 几何残差分支并叠加到偏移上 |
| `staas_strength` | `float` | `1.0` | `STAASv0` 几何偏移叠加权重 |
| `staas_tau0` / `staas_tau_min` / `staas_tau_max` | `float` | `0.02 / 0.005 / 0.08` | 透传到 `STAASv0` |
| `staas_fusion` | `bool` | `False` | 把 ST-AAS 几何统计拼接进神经偏移头特征 |
| `staas_v2_gate` | `bool` | `False` | 启用 ST-AAS v2 噪声/边缘条件门控 |
| `staas_v2_geo_weight` | `float` | `0.25` | ST-AAS v2 几何辅助偏移权重 |
| `staas_v2_gate_min` / `staas_v2_gate_max` | `float` | `0.0 / 1.0` | 残差门控的上下界 |
| `staas_v2_noise_ref_low` / `staas_v2_noise_ref_high` | `float` | `0.010 / 0.030` | 用尺度估计映射到 `[0, 1]` 噪声强度的两个参考点 |

偏移门控与裁剪：

| 参数 | 类型 | 默认值 | 说明 |
| ---- | ---- | ---- | ---- |
| `use_move_gate` | `bool` | `False` | 启用基于特征的逐点 sigmoid 门控 |
| `residual_clip` | `float` | `0.0` | 全局每点偏移 L2 范数裁剪上限，`0` 关闭 |
| `adaptive_clip` | `bool` | `False` | 启用按云尺度的分段自适应裁剪 |
| `adaptive_clip_min` | `float` | `0.006` | 自适应裁剪下界 |
| `adaptive_clip_max` | `float` | `0.020` | 自适应裁剪上界 |
| `adaptive_clip_ref_low` | `float` | `0.022` | 低噪参考尺度 |
| `adaptive_clip_ref_mid` | `float` | `0.030` | 中噪参考尺度 |
| `adaptive_clip_ref_high` | `float` | `0.040` | 高噪参考尺度 |
| `adaptive_clip_mid` | `float` | `0.010` | 中噪段裁剪目标值 |

噪声感知 / 混合分支：

| 参数 | 类型 | 默认值 | 说明 |
| ---- | ---- | ---- | ---- |
| `noise_aware_move_gate` | `bool` | `False` | 用每云 KNN 估计噪声尺度去缩放 `move_gate` |
| `noise_aware_gate_min` | `float` | `0.45` | 低噪云上的最小门控值 |
| `noise_aware_gate_ref_low` / `noise_aware_gate_ref_high` | `float` | `0.022 / 0.036` | 噪声感知门控的两个参考尺度 |
| `hybrid_safe_strong` | `bool` | `False` | 启用 safe（PW-SENEL 保护）/ strong（move_gate）双分支混合 |
| `hybrid_router_scale` | `float` | `1.0` | 混合分支路由强度 |

**`execute(noisy, return_offset=False)` 输入输出**

| 名称 | 类型 | 形状 | 含义 |
| ---- | ---- | ---- | ---- |
| 输入 `noisy` | `jt.Var` | `(B, N, 3)` | 含噪点云 |
| 输入 `return_offset` | `bool` | 标量 | 是否同时返回偏移张量 |
| 返回 `pred` | `jt.Var` | `(B, N, 3)` | 去噪后的点云，`pred = noisy + offset` |
| 返回 `offset`（可选） | `jt.Var` | `(B, N, 3)` | 残差偏移本体 |

#### 关键入口函数：`train` 与 `predict`

`ResidualDenoiser` 本身只是模型；项目把训练/推理流程包装成两个模块级函数，
通过命令行的 `--mode` 调用：

| 函数 | 行为 |
| ---- | ---- |
| `train(args)` | 构建 `ObjDenoiseDataset` 与 `ResidualDenoiser`、可选 `BatchPrefetcher`，执行 `args.steps` 步 Adam 训练，按 `args.save_every` 保存到 `args.ckpt`，并在 `args.ckpt.with_suffix('.train.csv')` 记录每步指标 |
| `predict(args)` | 加载 `args.ckpt`，遍历 `args.test_root` 下的 `shapenet/*/*/noisy.npy`，调用 `predict_points_in_chunks` 做分块推理，把结果写入 `args.out_dir/.../denoised.npy` |
| `predict_points_in_chunks(model, noisy_np, patch_size)` | 大点云的省显存推理：将 `noisy_np` 按 `patch_size` 切成连续 chunk 独立预测后再拼接，返回 `(N, 3) float32` 的 numpy 数组 |
| `make_zip(args)` / `validate_zip(args)` | 打包/校验提交 zip，详见 `scripts/predict.sh` |

**示例（直接调用模型）**

```python
import jittor as jt
from denoise_baseline import ResidualDenoiser

jt.flags.use_cuda = 1
model = ResidualDenoiser(k=16, feat_dim=256, hidden=256, use_pwsenel_v2=True)
model.load("experiments/denoise_pwsenel_v2/pwsenel_v2.pkl")
model.eval()

noisy = jt.array(noisy_np[None, ...])  # (1, N, 3)
pred = model(noisy)                    # (1, N, 3)
```

---

## 配置文件参数说明

### 配置文件结构

`configs/*.yaml` 是 `denoise_baseline.py` 的可复现实验入口。命令行参数与
YAML 字段一一对应（见 `apply_config` 与 `build_parser`）。一个完整配置由
四个顶层段组成：`experiment`、`paths`、`train`、`model`、`predict`。

### `experiment`

| 字段 | 含义 |
| ---- | ---- |
| `name` | 实验名，会出现在 run 目录、ckpt 摘要中 |
| `seed` | 全局随机种子，控制 NumPy/Python/Jittor RNG |

### `paths`

| 字段 | 含义 |
| ---- | ---- |
| `data_root` | 训练数据根目录（OBJ） |
| `test_root` | 测试数据根目录（含 `shapenet/<synset>/<model>/noisy.npy`） |
| `train_list` | 训练列表文件路径 |
| `out_dir` | `predict` 输出根目录 |
| `zip` | 提交 zip 文件路径 |
| `ckpt` | 训练保存的 checkpoint 路径，CSV 日志会写到同名 `.train.csv` |
| `warm_start` | 可选；训练前从兼容 checkpoint 加载已有权重 |

> 路径若是相对路径会在 `apply_config` 中相对仓库根目录解析。

### `train`

| 字段 | 默认/示例 | 含义 |
| ---- | ---- | ---- |
| `steps` | `5000` | 训练总步数 |
| `limit` | `2000` | 仅使用前 `limit` 个 OBJ 样本，`0` 不限 |
| `num_points` | `2048` | 每个样本采样点数 |
| `batch_size` | `2` | 每步样本数 |
| `lr` | `1e-4` | Adam 学习率 |
| `noise_min` / `noise_max` | `0.005 / 0.02` | 高斯噪声 σ 的随机均匀范围 |
| `cd_weight` | `0.1` | Chamfer 损失权重；smoke 配置中为 `0.0` |
| `identity_weight` | `0.0` | `pred ≈ noisy` 的恒等约束权重，用于低噪正则 |
| `movement_weight` | `0.0` | 偏移幅度正则 |
| `log_every` | `20` | 日志/CSV 写入周期 |
| `save_every` | `500` | checkpoint 保存周期 |
| `cache_clean` | `false` | 是否缓存归一化 clean 顶点（A6000 等大机型推荐 `true`） |
| `prefetch_workers` | `0` | 后台 NumPy batch worker 数；`>0` 启用 `BatchPrefetcher` |
| `prefetch_queue_size` | `4` | prefetch 队列大小 |
| `profile_times` | `false` | 记录 data/compute/step 计时 |
| `profile_system_every` | `0` | 每 N 个 logged step 打印一次 nvidia-smi/CPU 状态 |

### `model`

主干：

| 字段 | 含义 |
| ---- | ---- |
| `k` / `feat_dim` / `hidden` | 对应 `ResidualDenoiser(k, feat_dim, hidden)` |

研究开关（与 `ResidualDenoiser.__init__` 一一对应）：

| 字段 | 含义 |
| ---- | ---- |
| `pwsenel` | 启用 `PWSENEL` |
| `pwsenel_v2` / `pwsenel_v2_edge_lock` / `pwsenel_v2_gate_scale` | 启用并配置 `PWSENELv2Gate` |
| `staas` / `staas_strength` / `staas_tau0` / `staas_tau_min` / `staas_tau_max` | 启用并配置 `STAASv0` |
| `staas_fusion` | 把 ST-AAS 几何统计 fuse 进特征 |
| `staas_v2_gate` / `staas_v2_geo_weight` / `staas_v2_gate_min` / `staas_v2_gate_max` / `staas_v2_noise_ref_low` / `staas_v2_noise_ref_high` | ST-AAS v2 残差门控 |
| `move_gate` | 启用基于特征的偏移门控 |
| `residual_clip` | 全局偏移裁剪 |
| `adaptive_clip` 及 `adaptive_clip_*` 参数族 | 自适应裁剪 |
| `noise_aware_move_gate` 及 `noise_aware_gate_*` 参数族 | 噪声感知门控 |
| `hybrid_safe_strong` / `hybrid_router_scale` | safe/strong 混合路由 |

### `predict`

| 字段 | 含义 |
| ---- | ---- |
| `limit` | 只跑前 `limit` 个测试样本，`0` 不限 |
| `patch_size` | `predict_points_in_chunks` 的 chunk 大小 |

### `configs/profiles/` 的用途

`configs/profiles/*.yaml` 是**机器/算力相关的覆盖层**，通过
`bash scripts/train.sh <主配置> <profile>` 应用，或直接在命令行使用
`--profile <profile.yaml>`（可重复多次）。

合并规则在 `denoise_baseline.deep_update` 中实现：profile 只覆盖其中**显式
写出的字段**，其余字段沿用主配置；显式 CLI 参数继续覆盖前两者。

仓库现有 profile：

| 文件 | 典型用途 |
| ---- | ---- |
| `local_dev.yaml` | 本地开发机 smoke / 调试，小步数小 batch |
| `rtx4050.yaml` / `rtx5060ti.yaml` | 中等显存的桌面/游戏卡训练 |
| `a6000.yaml` | 48 GB 大显存正式训练，启用 `cache_clean` 与 prefetch |
| `a6000_fast_ablation.yaml` / `a6000_fast_move_gate.yaml` | A6000 上更激进的快速消融预设 |

> profile 设计目标是让同一个研究配置（如 `denoise_baseline.yaml`）能在
> 多种 GPU 上跑通而不改算法核心；如果新增机器，请新增独立 profile，而
> **不要**直接修改主研究配置。
