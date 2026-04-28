# 仓库结构与复现边界说明

本文档用于区分：哪些文件是复现代码和训练所必需的，哪些文件只是作者本人的实验流程、机器配置或开发辅助内容。

## 1. 复现必备文件

理解和运行正式赛点云降噪代码主要需要以下文件：

```text
denoise_baseline.py
requirements.txt
configs/denoise_baseline.yaml
configs/denoise_pwsenel.yaml
configs/denoise_staas_v0.yaml
scripts/env.sh
scripts/install_deps.sh
scripts/train.sh
scripts/predict.sh
scripts/check_env.py
scripts/check_data.py
README.md
README_CN.md
README_REPRODUCE.md
```

三个主要配置分别对应：

- `configs/denoise_baseline.yaml`：普通残差降噪 baseline
- `configs/denoise_pwsenel.yaml`：PW-SENEL 消融实验
- `configs/denoise_staas_v0.yaml`：ST-AAS v0 消融实验

## 2. 需要自行准备但不能上传 git 的外部文件

正式训练和预测需要官方数据集，但数据集不应提交到 git：

```text
dataset_train/
dataset_test_noisy/
```

这两个路径可以是实体目录，也可以是指向仓库外数据目录的软链接。

## 3. 官方 starter code

```text
starter_code/
```

该目录保存官方 starter code 和相关模块，尽量保持独立。作者自己的正式赛入口主要是：

```text
denoise_baseline.py
```

## 4. 作者机器 profile

```text
configs/profiles/local_dev.yaml
configs/profiles/rtx5060ti.yaml
configs/profiles/a6000.yaml
```

这些 profile 不属于核心算法，只用于记录和支持作者本人的硬件实验流程：

```text
local_dev：本地调试 / smoke test / 日志分析
rtx5060ti：第一次完整训练 / 中等规模消融预筛
a6000：更大规模正式训练
```

其他用户不需要拥有相同硬件。可以不传 profile 直接运行基础配置，也可以根据自己的 GPU、CPU、内存情况新建 profile。

## 5. Smoke test 与作者开发辅助脚本

```text
configs/denoise_smoke.yaml
run_config_smoke.sh
run_denoise_predict_smoke.sh
run_denoise_smoke.sh
run_train_baseline.sh
scripts/gpu_profile.py
scripts/freeze_env.sh
scripts/make_wheelhouse.sh
```

这些文件主要用于环境检查、快速冒烟测试、离线依赖准备、环境记录或作者本人的开发流程。它们有助于调试，但不是理解核心算法的必要文件。

## 6. 文档与项目元信息

```text
docs/GPU_PROFILES.md
docs/ST_AAS_CVMJ_PLAN.md
OPEN_SOURCE.md
LICENSE
NOTICE
CITATION.cff
```

说明：

- `docs/GPU_PROFILES.md`：解释 profile 机制和作者硬件示例
- `docs/ST_AAS_CVMJ_PLAN.md`：ST-AAS 后续设计笔记
- `LICENSE` / `NOTICE` / `CITATION.cff`：许可证、署名和引用信息

## 7. 明确不应上传的内容

以下内容不是复现源码，不应上传到 git：

```text
experiments/
results/
dataset_train/
dataset_test_noisy/
starter_code/.venv/
*.pkl
*.zip
*.npy
*.npz
*.log
.env
private keys / tokens
```

它们通常是本地数据、模型权重、训练日志、虚拟环境、预测结果、提交压缩包或私密凭据。
