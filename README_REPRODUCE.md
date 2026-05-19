# Reproduce: Formal Point-Cloud Denoising

目标：说明如何在配置好依赖和数据后复现本项目代码，并区分通用复现步骤与作者本人的实际实验流程。

仓库结构和复现边界详见：`docs/REPOSITORY_STRUCTURE_CN.md`（中文）/ `docs/REPOSITORY_STRUCTURE.md`（英文）。

## 目录约定

```text
/path/to/jittor-pointcloud-denoise
├── denoise_baseline.py          # 自写正式赛 baseline / PW-SENEL
├── configs/                     # 可复现实验配置
├── scripts/                     # 统一运行入口
├── experiments/                 # ckpt、run 归档、candidate registry
├── results/                     # predict 输出
├── dataset_train -> ...         # 训练数据软链接
└── dataset_test_noisy -> ...    # 测试数据软链接
```

## 环境

建议 Python 3.10~3.12，Jittor 可用 CUDA。若系统 Python 受 PEP668 管理，请使用 venv/conda，不要污染系统环境。

作者本地环境曾验证：

- Jittor CUDA OK
- gcc/g++-10 wrapper OK
- `trimesh`, `yaml`, `numpy` OK

通用复现步骤：

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
$PYTHON scripts/check_env.py --level jittor-cuda
```

如果只检查 Python / NumPy 工具链：

```bash
$PYTHON scripts/check_env.py --level python
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

可选：按显卡能力套 profile。profile 只调整 batch size、点数、训练步数等机器相关参数，不改变核心算法：

```bash
# 作者本地开发机示例
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

作者实际实验流程：

```text
本地开发机：代码编写、debug、小规模 smoke test、日志/结果分析
RTX 5060 Ti：第一次完整训练、中等规模实验、ablation 预筛
RTX A6000：更大规模正式训练、最终实验
```

这只是作者自己的操作流程，不是复现本代码的硬性硬件要求。其他用户可以根据自己的 GPU/CPU/内存情况修改 profile 或直接运行基础配置。

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

正式候选建议优先用统一推理入口，再做提交包校验：

```bash
source scripts/env.sh
$PYTHON scripts/unified_predict.py --name <candidate> --out-dir results/<candidate> --zip result_<candidate>.zip
$PYTHON scripts/check_submission.py result_<candidate>.zip --test-root dataset_test_noisy --require-float32
```

## Candidate Registry / Unified Pipeline

正式候选要求候选记录自动化、官方 `evaluate.py` 口径可追踪、A/B 同构入口固化。相关说明见：`docs/CANDIDATE_REGISTRY.md`。

当前推荐候选推理入口是统一脚本，而不是零散预测脚本。下面是入口形态示例，不代表推荐具体算法：

```bash
source scripts/env.sh
$PYTHON scripts/unified_predict.py \
  --name <candidate> \
  --patch-size 8192 \
  --out-dir results/<candidate> \
  --zip result_<candidate>.zip
$PYTHON scripts/check_submission.py result_<candidate>.zip \
  --test-root dataset_test_noisy \
  --require-float32
```

历史 `router_t0165` / hard-router 示例只作为 Phase-1 记录保留，不是当前推荐候选。当前 registry anchor 是 `blend_best075_lir025_20260517` / fixed075，官方分数 `53.32`。GARA-D identity / bounded tiny 仍是 experimental，tiny shrink 不是官方提交推荐。

候选登记示例：

```bash
$PYTHON scripts/candidate_registry.py \
  --name <candidate> \
  --config <config.yaml> \
  --ckpt <checkpoint.pkl> \
  --zip result_<candidate>.zip \
  --branch "<pipeline-or-artifact-branch>" \
  --patch-size 8192 \
  --submission-check passed \
  --conclusion "<有证据来源的结论>"
```

候选输出：

- `experiments/candidates.jsonl`
- `experiments/candidate_registry.csv`
- `experiments/candidate_registry.md`

精选的文字摘要会另外保存在 `docs/experiments/`，用于快速阅读；完整工作区产物仍留在 `analysis/`，不要把它当作主线复现入口。

## 当前模型

- `ResidualDenoiser`: `FeatureExtraction/EdgeConv + offset head`
- 预测形式：`pred_clean = noisy + offset`
- `PWSENEL`: `PeiWei Softmax Edge-aware Noise Elimination and Locking`
  - Softmax 邻域筛噪
  - MaxPool 边缘锁定
  - 可通过配置 `model.pwsenel: true/false` 做 ablation

## 注意

当前推理使用 contiguous chunk 分块，优先保证不爆显存；后续应升级为 FPS/KNN overlapping patch + weighted stitching，以提升大点云质量。
