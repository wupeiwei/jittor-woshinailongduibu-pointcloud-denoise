# Reproduce: Formal Point-Cloud Denoising

目标：让代码能在本机开发，也能迁移到 A6000 训练机复现。

## 目录约定

```text
/home/sallen/jittor-pointcloud-denoise
├── denoise_baseline.py          # 自写正式赛 baseline / PW-SENEL
├── configs/                     # 可复现实验配置
├── scripts/                     # 统一运行入口
├── experiments/                 # ckpt 与 run 归档
├── results/                     # predict 输出
├── dataset_train -> ...         # 训练数据软链接
└── dataset_test_noisy -> ...    # 测试数据软链接
```

## 环境

建议 Python 3.10~3.12，Jittor 可用 CUDA。若系统 Python 受 PEP668 管理，请使用 venv/conda，不要污染系统环境。

本机已验证：

- Jittor CUDA OK
- gcc/g++-10 wrapper OK
- `trimesh`, `yaml`, `numpy` OK

A6000 训练机建议：

1. 克隆/复制本项目。
2. 准备数据目录，并让配置里的 `paths.data_root` / `paths.test_root` 指向真实位置。
3. 准备 Python 环境并安装依赖，优先用项目内统一依赖清单：

```bash
cd /path/to/jittor-pointcloud-denoise
source scripts/env.sh
bash scripts/install_deps.sh
```

如果服务器只有 `python3` 没有 `python`，不用改脚本；`scripts/env.sh` 会自动设置 `PYTHON=python3`。也可以手动指定：

```bash
PYTHON=python3 bash scripts/install_deps.sh
```

4. 如 Jittor 编译失败，安装 gcc-10/g++-10，并让 `gcc/g++` 指向 10.x，或修改 `scripts/env.sh`。
5. 运行环境检查：

```bash
cd /path/to/jittor-pointcloud-denoise
source scripts/env.sh
$PYTHON scripts/check_env.py
```

## 训练

最小冒烟测试，仅用于检查环境/数据/日志/ckpt，不用于比较分数：

```bash
bash scripts/train.sh configs/denoise_smoke.yaml
```

普通 baseline，正式训练配置，默认使用 `offset_mse + 0.1 * Chamfer`：

```bash
bash scripts/train.sh configs/denoise_baseline.yaml
```

PW-SENEL ablation，与 baseline 保持同一 loss 权重：

```bash
bash scripts/train.sh configs/denoise_pwsenel.yaml
```

按显卡能力套 profile：

```bash
# 本机/开发机
bash scripts/train.sh configs/denoise_baseline.yaml configs/profiles/local_dev.yaml

# RTX 5060 Ti
bash scripts/train.sh configs/denoise_baseline.yaml configs/profiles/rtx5060ti.yaml

# RTX A6000
bash scripts/train.sh configs/denoise_baseline.yaml configs/profiles/a6000.yaml
```

自动查看推荐 profile：

```bash
source scripts/env.sh
$PYTHON scripts/gpu_profile.py
```

如果服务器不能联网，可以先在能联网且 Python 版本/系统接近的机器制作离线依赖包：

```bash
bash scripts/make_wheelhouse.sh
```

然后把 `wheelhouse.tar.gz` 带到训练机，解压后安装：

```bash
tar -xzf wheelhouse.tar.gz
WHEELHOUSE=wheelhouse bash scripts/install_deps.sh
```

如果服务器只有 `python3` 没有 `python`，不用改脚本；`scripts/env.sh` 会自动设置 `PYTHON=python3`。也可以手动指定：

```bash
PYTHON=python3 bash scripts/train.sh configs/denoise_baseline.yaml configs/profiles/rtx5060ti.yaml
```

每次训练会保存：

- `experiments/runs/<name_timestamp>/config.yaml`
- `experiments/runs/<name_timestamp>/env.txt`
- `experiments/runs/<name_timestamp>/meta.txt`
- `experiments/runs/<name_timestamp>/train.log`
- ckpt 路径由配置 `paths.ckpt` 控制

## 推理和打包

```bash
bash scripts/predict.sh configs/denoise_baseline.yaml
```

输出 zip 路径由配置 `paths.zip` 控制，结构应为：

```text
shapenet/<synset_id>/<model_id>/denoised.npy
```

## 当前模型

- `ResidualDenoiser`: `FeatureExtraction/EdgeConv + offset head`
- 预测形式：`pred_clean = noisy + offset`
- `PWSENEL`: `PeiWei Softmax Edge-aware Noise Elimination and Locking`
  - Softmax 邻域筛噪
  - MaxPool 边缘锁定
  - 可通过配置 `model.pwsenel: true/false` 做 ablation

## 注意

当前推理使用 contiguous chunk 分块，优先保证不爆显存；后续应升级为 FPS/KNN overlapping patch + weighted stitching，以提升大点云质量。
