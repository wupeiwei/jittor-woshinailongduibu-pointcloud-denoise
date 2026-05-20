# 质量闸门 (Quality Gate)

## 概述

提交前必须通过的所有质量检查清单。本文档基于仓库中实际的验证脚本和测试代码整理而成，确保每次提交包在发送到官方评测前已通过本地所有硬门槛。

---

## 1. 必跑测试

项目使用 `pytest`，测试根目录为 `tests/`，配置位于 `pytest.ini`。已注册 marker：
- `gpu`：需要 GPU 的测试（在无 GPU 环境通过 `-m "not gpu"` 跳过）

### 1.1 本机可运行（无 GPU 要求）

以下测试仅需 Jittor CPU 后端（`conftest.py` 自动设置 `nvcc_path=""` 强制 CPU 模式）：

```bash
pytest tests/ -m "not gpu"
```

具体包括：

| 测试文件 | 覆盖范围 |
|---------|---------|
| `tests/unit/test_denoise_utils.py` | `gather_neighbors` 函数的 shape、数值正确性、边界情况（K=1/B=1）|
| `tests/unit/test_dataset.py` | `ObjDenoiseDataset` 初始化、datalist 解析、缓存、getitem 输出 shape/dtype |
| `tests/unit/test_models.py` 中未标记 `@pytest.mark.gpu` 的用例 | `PWSENEL` forward shape、`STAASv0` forward shape 及 stats 字段 |

### 1.2 必须 A6000 运行

以下测试/验证 **必须** 在有 CUDA GPU（推荐 A6000 48GB）的环境执行：

```bash
pytest tests/ -m "gpu"
```

具体包括：

| 测试文件 | 覆盖范围 |
|---------|---------|
| `tests/unit/test_models.py` 中 `@pytest.mark.gpu` 用例 | `PWSENELv2Gate` forward shape、gate 值域 [0,1]、return_stats 字段完整性 |
| `tests/integration/test_predict_pipeline.py`（整个文件标记 gpu）| `ResidualDenoiser` 完整端到端前向：baseline/offset pair/pwsenel+move_gate/staas_fusion |

此外，以下 **非 pytest** 验证也需在 A6000 执行：
- 完整 200 样本推理 (`predict`) + 打包 (`make_zip`) + 包校验 (`check_submission`)
- 大规模点云（50000 点/样本）推理稳定性和内存压力测试
- 使用 `configs/profiles/a6000.yaml` profile 运行正式训练/推理流程

---

## 2. 提交包 (zip) 验证

验证脚本：`scripts/check_submission.py`

### 2.1 格式要求

根据 `check_submission.py` 中 `validate_zip()` 函数的检查逻辑，提交 zip 必须满足：

| 检查项 | 要求 |
|--------|------|
| zip 可读 | 文件存在且为合法 zip |
| 条目路径格式 | 固定 4 级：`shapenet/<category>/<model>/denoised.npy` |
| 文件数量 | 默认要求 **200** 个文件（`--expected-count 200`）|
| 每个 npy 的 shape | 默认 `(50000, 3)`（`--expected-shape 50000,3`）|
| ndim 和最后一维 | 必须为 2D 且 shape[-1] == 3 |
| dtype | 必须为数值类型（`np.issubdtype(arr.dtype, np.number)`）；如加 `--require-float32` 则必须 float32 |
| 有限值 | 所有元素必须是有限数（无 NaN/Inf）|
| 无重复条目 | zip 内不可有重复文件名 |
| test_root 匹配 | 若提供 `--test-root`，则 zip 条目必须与 `test_root` 下的 `noisy.npy` 目录结构一一对应 |

### 2.2 验证命令

```bash
# 基本验证（无 test_root 对照）
python scripts/check_submission.py result_xxx.zip \
  --expected-count 200 \
  --expected-shape 50000,3 \
  --require-float32 \
  --no-test-root-match

# 完整验证（含 test_root 条目匹配）
python scripts/check_submission.py result_xxx.zip \
  --config configs/denoise_baseline.yaml \
  --expected-count 200 \
  --expected-shape 50000,3 \
  --require-float32 \
  --test-root /path/to/dataset_test_noisy

# 使用 config 中的 paths.zip 和 paths.test_root
python scripts/check_submission.py --config configs/denoise_baseline.yaml --profile configs/profiles/a6000.yaml
```

`denoise_baseline.py` 中也提供轻量校验入口：
```bash
python denoise_baseline.py validate_zip --zip result_xxx.zip
```
注意：这只做路径格式检查，**不** 验证 shape/dtype/finite，正式提交前必须使用 `scripts/check_submission.py`。

### 2.3 常见失败原因

| 失败信息 | 原因 | 修复方法 |
|---------|------|---------|
| `file count mismatch` | 推理未覆盖全部测试样本，或 `--limit` 意外截断 | 确认 `predict` 未设 limit；检查 test_root 下 noisy.npy 数量 |
| `bad path: xxx` | 目录层级多了/少了（如额外嵌套 `dataset_test_noisy`）| 检查 `out_dir` 结构，确保是 `shapenet/<cat>/<model>/denoised.npy` |
| `bad shape: ... got (N, 3), expected (50000, 3)` | 推理未保持原始点数 | 检查 `predict_points_in_chunks` 输出拼接逻辑 |
| `NaN/Inf found` | 模型数值溢出 | 检查 residual clip / adaptive clip 设置；排查 Jittor JIT 编译问题 |
| `non-float32 dtype` | 模型输出或 numpy 保存时精度不对 | 确保 `.astype(np.float32)` 后再 `np.save` |
| `missing entries vs test_root` | zip 中缺少某些样本 | 对比 zip 内容和 test_root 目录 |
| `extra entries vs test_root` | zip 中包含不应存在的文件 | 清理 out_dir 中的残留文件后重新打包 |
| `duplicate entries` | zip 内有重复文件名 | 检查打包脚本是否多次写入同一相对路径 |

---

## 3. Candidate Registry 验证

Registry 由 `scripts/candidate_registry.py` 维护，输出三份同步文件：
- `experiments/candidates.jsonl` — 机器可读追加日志
- `experiments/candidate_registry.csv` — 电子表格友好格式
- `experiments/candidate_registry.md` — 人类可读摘要

### 3.1 必填字段

Registry schema 版本为 `candidate-registry-v1`，关键字段：

| 字段 | 说明 | 是否必填 |
|------|------|---------|
| `schema_version` | 固定为 `candidate-registry-v1` | 必填 |
| `name` | 候选唯一标识符 | 必填 |
| `stage` | 阶段标识（如 `Phase 1`, `Phase 2`, `official-full`）| 必填 |
| `status` | 状态（`candidate` / `submitted` / `baseline` / `archive` / `rejected`）| 必填 |
| `created_at` | ISO 8601 时间戳 | 必填 |
| `config_path` | 使用的配置文件路径 | 必填 |
| `checkpoint_path` | 模型权重路径 | 必填 |
| `zip_path` | 提交包路径 | 必填 |
| `zip_sha256` | 提交包 SHA256 校验和 | 必填 |
| `submission_check` | `check_submission.py` 的结果（`passed` 或详细信息）| 必填 |
| `conclusion` | 结论说明 | 必填 |
| `evaluate_py_path` | 使用的评估脚本路径 | 建议填写 |
| `evaluate_py_sha256` | 评估脚本 SHA256 | 建议填写 |
| `official_submitted` | 是否已提交到官方（`true`/`false`）| 必填 |
| `official_score_recorded` | 官方分数是否已记录（`true`/`false`）| 必填 |

### 3.2 一致性检查

`candidates.jsonl` 与 `candidate_registry.csv` 之间的一致性要求：

1. **行数一致**：JSONL 每行对应 CSV 的一行数据（不含表头）
2. **字段对齐**：两者字段名和顺序需与 `FIELDS` 列表一致
3. **SHA256 可验证**：若 `zip_path` 指向的文件仍存在，其 SHA256 应与记录匹配
4. **状态流转合法**：`candidate` → `submitted` → 记录 `official_score`；或 → `archive`/`rejected`
5. **名称唯一性**：同一 `name` + `stage` 组合如果出现多次（如先 candidate 后 submitted），应有明确的状态演进

### 3.3 验证命令

```bash
# 检查 JSONL 文件格式完整性（每行可解析为 JSON，字段覆盖 FIELDS）
python -c "
import json
from pathlib import Path
fields = set(json.loads(Path('experiments/candidates.jsonl').read_text().splitlines()[0]).keys())
for i, line in enumerate(Path('experiments/candidates.jsonl').read_text().splitlines(), 1):
    row = json.loads(line)
    assert 'schema_version' in row, f'line {i}: missing schema_version'
    assert 'name' in row, f'line {i}: missing name'
    assert 'zip_sha256' in row, f'line {i}: missing zip_sha256'
print(f'OK: {i} entries validated')
"

# 检查 CSV 行数与 JSONL 一致
python -c "
from pathlib import Path
jsonl_lines = len(Path('experiments/candidates.jsonl').read_text().strip().splitlines())
csv_lines = len(Path('experiments/candidate_registry.csv').read_text().strip().splitlines()) - 1  # 减去表头
assert jsonl_lines == csv_lines, f'mismatch: JSONL={jsonl_lines} CSV={csv_lines}'
print(f'OK: {jsonl_lines} entries in sync')
"
```

---

## 4. Hidden Leak 禁令

### 4.1 什么是 Hidden Leak

Hidden Leak 指在推理（predict）阶段使用了 **隐藏测试集不提供** 的信息来决定输出。由于竞赛 B 榜评测时只提供 `noisy.npy`（带噪声的点云），任何依赖其他信息的推理逻辑都将在正式环境中失败或构成作弊。

### 4.2 禁止在推理中使用的信息

| 禁止使用的信息 | 说明 |
|---------------|------|
| GT 坐标（ground truth 点云）| 隐藏测试不提供干净点云 |
| sigma（噪声标准差参数）| 隐藏测试不提供噪声级别标签 |
| CD/P2S 分数 | 这是评测输出，不是推理输入 |
| 训练时的标签信息 | 任何训练时才有的 clean/GT 标注 |
| 官方 leaderboard 反馈 | 不可将排名/分数作为推理条件 |
| `datalist` 中的类别标签 | 推理时只能根据输入 noisy 点云本身做决策 |

### 4.3 如何自检

**代码层面排查**：

```bash
# 搜索推理路径中是否引用了 GT 相关变量
grep -rn "clean\|ground_truth\|gt_points\|sigma" scripts/unified_predict.py denoise_baseline.py \
  | grep -v "^#\|^.*#\|noqa\|comment\|doc\|test"

# 搜索推理路径中是否加载了不该加载的文件
grep -rn "clean\.npy\|gt\.npy\|sigma\|label" scripts/unified_predict.py denoise_baseline.py
```

**逻辑层面自检清单**：

1. `predict()` 函数中只读取 `noisy.npy`，不读取同目录下的其他文件
2. 路由/门控决策仅基于 noisy 点云自身的几何统计（如平面残差 p75）
3. 不使用 `datalist` 文件中的 category 信息做条件分支（除非只是路径遍历）
4. 后处理（如 LIR、adaptive blend）只使用 noisy 和模型输出，不参考 GT

**已有安全注释**（代码中已标注）：

`scripts/unified_predict.py` 中明确注释：
> "隐藏测试没有 GT，只能用 noisy 自身估计难度"
> "不读取 GT，也不依赖 leaderboard 反馈，因此可以用于隐藏测试风险排查"

### 4.4 已知安全模式

以下用法在推理中是 **允许的**：

| 安全模式 | 说明 |
|---------|------|
| noisy 点云几何统计 | 如 KNN 距离、平面残差 p75、局部方差等 —— 仅从 noisy.npy 计算 |
| 预训练模型权重 | 训练时用了 GT（正常监督学习），推理时只用模型参数 |
| 固定超参数 | 如 router_threshold、gate_temperature —— 在代码/config 中固定，不依赖测试集 |
| zip-level 插值/混合 | 两个已生成 zip 的线性组合 —— 不使用任何 GT 信息 |
| noisy-only adaptive blend | 根据 noisy 的几何指标决定混合比例 —— 如 `noise_gate` 策略 |

---

## 5. 提交前 Checklist

提交到官方评测前，逐项确认：

- [ ] **pytest 全部通过**
  ```bash
  pytest tests/ -m "not gpu"          # 本机
  pytest tests/ -m "gpu"              # A6000（若有）
  ```

- [ ] **完整推理无错误**
  - 200 个样本全部生成 `denoised.npy`
  - 无 NaN/Inf 警告
  - 输出 shape 为 `(50000, 3)`

- [ ] **check_submission 通过**
  ```bash
  python scripts/check_submission.py result_xxx.zip \
    --expected-count 200 --expected-shape 50000,3 \
    --require-float32 --test-root /path/to/dataset_test_noisy
  ```

- [ ] **zip SHA256 记录**
  ```bash
  sha256sum result_xxx.zip
  ```

- [ ] **Candidate Registry 已更新**
  - `experiments/candidates.jsonl` 追加新条目
  - `experiments/candidate_registry.csv` 同步
  - 所有必填字段完整

- [ ] **Hidden Leak 自检通过**
  - 推理路径未引用 GT/sigma/CD 分数
  - 路由决策仅基于 noisy 几何统计
  - 后处理无泄漏

- [ ] **Git 状态干净**（建议）
  ```bash
  git status          # 确认无未提交的关键改动
  git log -1          # 记录当前 commit hash 到 registry
  ```

- [ ] **Profile 配置一致**
  - 使用的 profile（如 `configs/profiles/a6000.yaml`）与 registry 记录一致
  - `patch_size`、`chunk_size`、`stitching_strategy` 与实际推理参数一致

- [ ] **官方 evaluate.py 版本确认**
  ```bash
  sha256sum starter_code/evaluate.py
  # 应匹配 registry 中的 evaluate_py_sha256
  ```

- [ ] **每日提交配额检查**
  - 确认当日还有剩余提交次数
  - 记录提交时间到 registry notes
