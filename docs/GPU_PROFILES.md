# GPU Profiles and Migration Notes

目标：同一份代码能在本机、RTX 5060 Ti、双 A6000 上稳定迁移。

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
```

注意：5060 Ti 可能有不同显存版本。如果是 8GB，降低 `batch_size` 或 `num_points`；如果是 16GB，可保持 profile 默认值。

### RTX A6000 / 双 A6000

用途：

- 大规模正式训练
- 最终 ablation
- 最终预测和提交包生成

命令：

```bash
bash scripts/train.sh configs/denoise_baseline.yaml configs/profiles/a6000.yaml
bash scripts/train.sh configs/denoise_pwsenel.yaml configs/profiles/a6000.yaml
```

当前先保证单卡 A6000 可复现。双卡训练后续再做，不要一开始就把分布式复杂度混进去。

## 自动查看推荐 profile

```bash
source scripts/env.sh
$PYTHON scripts/gpu_profile.py
```

## 迁移前检查

```bash
source scripts/env.sh
$PYTHON scripts/check_env.py
```

必须确认：

- 能看到目标 GPU
- Jittor CUDA smoke test 成功
- gcc/g++ 可用
- 数据路径存在

## Profile 设计原则

- `configs/denoise_baseline.yaml` / `configs/denoise_pwsenel.yaml` 保存实验本体。
- `configs/profiles/*.yaml` 只覆盖机器能力相关参数，例如：
  - `batch_size`
  - `num_points`
  - `patch_size`
  - `steps`
  - `limit`
- 不要在 profile 里改模型创新逻辑，避免 ablation 混乱。

## 显存调参优先级

爆显存时按顺序降：

1. `train.batch_size`
2. `train.num_points`
3. `model.k`
4. `model.feat_dim`
5. `predict.patch_size`

不要第一反应改模型结构，否则结果不可比。
