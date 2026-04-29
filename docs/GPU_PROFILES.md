# GPU Profiles and Migration Notes

目标：同一份代码能在开发机、RTX 5060 Ti、RTX A6000 上稳定迁移。这里的 profile 是作者工作流预设，不是复现者的硬性环境要求。

## 基本原则

- 先保证单卡可复现，再考虑双卡或更复杂的并行训练。
- 所有训练/预测都优先走 `source scripts/env.sh` 后的 `$PYTHON`，不要直接假设系统 `python3` 可用。
- `configs/profiles/*.yaml` 只覆盖机器能力相关参数，例如 `batch_size`、`num_points`、`patch_size`、`steps`、`limit`。
- 不要在 profile 里改创新模块开关，避免 ablation 混乱；创新开关放在 `configs/denoise_*.yaml`。
- 爆显存时先降资源参数，不要第一反应改模型逻辑。

## 首次迁移检查

在 5060Ti / A6000 上 clone 或拷贝项目后，建议先跑：

```bash
cd jittor-pointcloud-denoise
source scripts/env.sh
bash scripts/install_deps.sh
$PYTHON scripts/check_env.py
$PYTHON scripts/check_data.py --limit 5
$PYTHON scripts/gpu_profile.py
```

如果 PyPI 网络慢，可使用清华源：

```bash
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple bash scripts/install_deps.sh
```

如果有离线 wheelhouse：

```bash
WHEELHOUSE=/path/to/wheelhouse bash scripts/install_deps.sh
```

## CuPy 要求

Jittor CUDA 训练在目标机器上可能会导入 CuPy。缺少 CuPy 时不要试图用 `.data`、`.numpy()` 或 `.item()` 绕过；应安装匹配 CUDA 的 wheel。

CUDA 12.x 推荐：

```bash
$PYTHON -m pip install "cupy-cuda12x>=13.0,<14.0"
```

CUDA 11.x 机器则安装对应的 `cupy-cuda11x`。

## 推荐角色分工

### 本机 / 低显存开发机

用途：

- 改代码
- 小样本 smoke test
- debug 数据读取和提交格式
- 不承担正式大规模训练

命令：

```bash
bash scripts/train.sh configs/denoise_baseline.yaml configs/profiles/local_dev.yaml
```

### RTX 5060 Ti

用途：

- 中等规模训练
- ablation 预筛
- 验证 loss 是否正常下降
- 在上 A6000 前排除代码 bug

命令：

```bash
bash scripts/train.sh configs/denoise_baseline.yaml configs/profiles/rtx5060ti.yaml
bash scripts/train.sh configs/denoise_pwsenel.yaml configs/profiles/rtx5060ti.yaml
bash scripts/train.sh configs/denoise_staas_v0.yaml configs/profiles/rtx5060ti.yaml
```

注意：5060 Ti 可能有不同显存版本。如果是 8GB，优先降低 `batch_size` 或 `num_points`；如果是 16GB，可保持 profile 默认值。

### RTX A6000 / 双 A6000

用途：

- 大规模正式训练
- 最终 ablation
- 最终预测和提交包生成

命令：

```bash
bash scripts/train.sh configs/denoise_baseline.yaml configs/profiles/a6000.yaml
bash scripts/train.sh configs/denoise_pwsenel.yaml configs/profiles/a6000.yaml
bash scripts/train.sh configs/denoise_staas_v0.yaml configs/profiles/a6000.yaml
```

当前先保证单卡 A6000 可复现。双卡训练后续再做，不要一开始就把分布式复杂度混进去。

## 自动查看推荐 profile

```bash
source scripts/env.sh
$PYTHON scripts/gpu_profile.py
```

## 显存调参优先级

爆显存时按顺序降：

1. `train.batch_size`
2. `train.num_points`
3. `model.k`
4. `model.feat_dim`
5. `predict.patch_size`

不要第一反应改模型结构，否则结果不可比。
