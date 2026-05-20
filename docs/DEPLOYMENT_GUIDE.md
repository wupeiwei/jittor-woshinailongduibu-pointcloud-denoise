# 部署指南

本指南面向首次将 `jittor-pointcloud-denoise` 部署到一台新机器（本地或服务器）的用户，覆盖
环境准备、数据挂载、训练、推理和常见问题排查。命令均假设仓库已经被克隆/同步到
`/path/to/jittor-pointcloud-denoise`，并在该目录下执行。

## 1. 环境要求

- **操作系统**：Linux（Ubuntu 22.04 / 24.04 已验证）。其他发行版只要能装 CUDA 12.x 与 Python 3.10–3.12 也可。
- **Python**：3.10、3.11 或 3.12。强烈建议使用 venv / conda 隔离，不要污染系统 Python（受 PEP 668 管理的发行版会直接拒绝 `pip install`）。
- **CUDA / GPU**：
  - 至少需要一张支持 CUDA 12.x 的 NVIDIA GPU（如 RTX 4050、RTX 5060 Ti、RTX A6000 等）。
  - 必须安装与 GPU 匹配的 NVIDIA 驱动；`cupy-cuda12x` 仅支持 CUDA 12.x 运行时。
  - 显存建议 ≥ 8 GB（小数据消融）或 ≥ 24 GB（正式训练）。
  - CPU 模式可通过 `--cpu` 启动，仅用于本机 smoke。
- **编译工具链**：Jittor 需要 C++ 编译器。推荐 `gcc-10`/`g++-10`；仓库提供
  `starter_code/.toolchain-gcc10` 作为可选 wrapper，由 `scripts/env.sh` 自动加入 `PATH`。
- **磁盘空间**：训练数据（ShapeNet 子集）+ 实验产物建议预留 ≥ 20 GB。

## 2. 环境准备

### 2.1 创建虚拟环境

仓库 `scripts/env.sh` 默认会激活 `starter_code/.venv`（如存在）。最稳的做法是把
虚拟环境创建到这个路径，让 `env.sh` 自动激活：

```bash
cd /path/to/jittor-pointcloud-denoise
python3 -m venv starter_code/.venv
source starter_code/.venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

如果想把虚拟环境放在别处，导出 `VENV_PATH` 即可：

```bash
export VENV_PATH=/path/to/your/venv
```

### 2.2 安装依赖（来自 `requirements.txt`）

仓库提供一键安装脚本，会自动 source `env.sh`、安装 `requirements.txt`，并补装 CuPy：

```bash
cd /path/to/jittor-pointcloud-denoise
source scripts/env.sh
bash scripts/install_deps.sh
```

`requirements.txt` 包含的核心依赖：

| 依赖 | 版本约束 | 说明 |
| ---- | ---- | ---- |
| `jittor` | `>=1.3.9` | 训练/推理深度学习框架 |
| `cupy-cuda12x` | `>=13.0,<14.0` | Jittor CUDA 同步/反向所需；CuPy 14 会拉 numpy ≥ 2，与本项目冲突 |
| `numpy` | `>=1.23,<2.0` | 主版本锁在 1.x |
| `trimesh` | `>=4.0` | 加载 ShapeNet OBJ |
| `PyYAML` | `>=6.0` | 读取配置 |
| `scipy` | `>=1.10` | 数值与 KNN 等辅助计算 |
| `omegaconf` | `>=2.3` | 配置工具（部分 analysis/script 使用） |
| `tqdm` | `>=4.60` | 进度条 |
| `pytest` / `pytest-cov` | `>=7.0` / `>=4.0` | 测试，可选 |

如果服务器无外网，可在能联网的同系统机器上执行：

```bash
bash scripts/make_wheelhouse.sh
```

把生成的 `wheelhouse.tar.gz` 拷贝到目标机解压，再这样安装：

```bash
tar -xzf wheelhouse.tar.gz
WHEELHOUSE=wheelhouse bash scripts/install_deps.sh
```

或者通过私有 PyPI 镜像：

```bash
PIP_INDEX_URL=https://your-mirror/simple/ \
PIP_TRUSTED_HOST=your-mirror \
bash scripts/install_deps.sh
```

如果只有 `python3` 没有 `python`，无需改脚本，`scripts/env.sh` 会自动设定 `PYTHON=python3`；
也可以手动覆盖：

```bash
PYTHON=python3 bash scripts/install_deps.sh
```

### 2.3 设置环境变量（来自 `scripts/env.sh`）

每次进入新 shell 都需要先 source：

```bash
cd /path/to/jittor-pointcloud-denoise
source scripts/env.sh
```

它会做以下事情：

- 推导 `PROJECT_ROOT` 为仓库根目录，并 `cd` 进去；
- 若 `starter_code/.toolchain-gcc10/` 存在，把它加入 `PATH`，让 Jittor 选 gcc/g++-10；
- 设置 `CC=gcc`、`CXX=g++`（可外部覆盖）；
- 设 `DISABLE_MULTIPROCESSING=1`，关闭 Jittor 多进程编译，避免不稳定；
- 设 `JITTOR_HOME=$PROJECT_ROOT/.jittor_home`，把 Jittor 缓存收敛到仓库内；
- 激活 `VENV_PATH`（默认 `starter_code/.venv`）；
- 导出 `PYTHON`，后续脚本统一用 `$PYTHON` 调用解释器。

完成后跑一次环境检查：

```bash
$PYTHON scripts/check_env.py --level jittor-cuda
```

仅校验 Python/NumPy 工具链可用：

```bash
$PYTHON scripts/check_env.py --level python
```

## 3. 数据准备

### 3.1 数据集格式

- **训练集**：ShapeNet 子集，每个模型是 OBJ 网格，路径形如：

  ```text
  <data_root>/shapenet/<synset_id>/<model_id>/models/model_normalized.obj
  ```

  训练列表 `starter_code/datalist/train.txt` 每行一个相对路径，可以是 `shapenet/<synset>/<model>` 或仅 `<synset>/<model>`，两种格式 `ObjDenoiseDataset` 都兼容。

- **测试集**：含噪点云预先以 numpy 保存：

  ```text
  <test_root>/shapenet/<synset_id>/<model_id>/noisy.npy
  ```

  `noisy.npy` 是 `(N, 3)` 的 `float32` 数组。

- **预测输出**：与测试输入一一对应：

  ```text
  <out_dir>/shapenet/<synset_id>/<model_id>/denoised.npy
  ```

  最终打成 zip 提交。

### 3.2 推荐目录结构

仓库根目录推荐保持下述结构（数据用软链接挂入）：

```text
/path/to/jittor-pointcloud-denoise
├── denoise_baseline.py
├── configs/
├── scripts/
├── experiments/
├── results/
├── dataset_train -> /mnt/data/shapenet_obj
└── dataset_test_noisy -> /mnt/data/shapenet_test_noisy
```

### 3.3 配置数据路径

- **方法 A：软链接（推荐）**：

  ```bash
  ln -s /mnt/data/shapenet_obj        dataset_train
  ln -s /mnt/data/shapenet_test_noisy dataset_test_noisy
  ```

  `configs/*.yaml` 默认就指向 `dataset_train` / `dataset_test_noisy`，不用改任何文件。

- **方法 B：修改配置**：在自定义配置里覆盖 `paths.data_root` / `paths.test_root` 为绝对路径。
  `apply_config` 会自动把仓库相对路径解析成绝对路径，绝对路径会原样保留。

- **方法 C：命令行覆盖**：训练或推理时追加 `--data-root /abs/path` 等参数，会覆盖
  YAML 默认值（CLI 优先级最高）。

数据是否就绪可以用：

```bash
$PYTHON scripts/check_data.py --config configs/denoise_baseline.yaml --limit 3
```

## 4. 训练流程

`scripts/train.sh` 是统一训练入口，会自动建立 `experiments/runs/<exp>_<timestamp>/`
目录，复制配置、抓取环境信息、抽样校验数据，再执行 `denoise_baseline.py --mode train`：

```bash
# 最小冒烟（仅校验环境/数据/日志/ckpt 流程是否打通，不用于评分）
bash scripts/train.sh configs/denoise_smoke.yaml

# 正式 baseline（offset_mse + 0.1 * Chamfer）
bash scripts/train.sh configs/denoise_baseline.yaml

# 配合机器 profile 使用
bash scripts/train.sh configs/denoise_baseline.yaml configs/profiles/local_dev.yaml
bash scripts/train.sh configs/denoise_baseline.yaml configs/profiles/rtx5060ti.yaml
bash scripts/train.sh configs/denoise_baseline.yaml configs/profiles/a6000.yaml
```

### 4.1 配置文件选择说明

仓库 `configs/` 中的配置按用途分类：

| 类型 | 代表 | 用途 |
| ---- | ---- | ---- |
| Smoke | `denoise_smoke.yaml` | 极少步数/极少样本，仅做流水线校验 |
| 正式 baseline | `denoise_baseline.yaml` | 默认对比基准，约 5000 步 |
| 算子消融 | `denoise_pwsenel_v2*.yaml`、`denoise_staas_v*.yaml`、`denoise_move_gate.yaml`、`denoise_noise_aware_move_gate*.yaml`、`denoise_hybrid_safe_strong_v3.yaml` 等 | 启用对应研究模块做消融 |
| Profiles | `configs/profiles/*.yaml` | 机器相关覆盖（batch、点数、步数、prefetch 等），不改算法 |

合并优先级（来自 `apply_config`）：**主配置 < profile（多个按顺序合并） < 显式 CLI 参数**。
profile 只覆盖它显式列出的字段。

如果只想暂时调一两个参数，无需新建 YAML，可以直接 CLI 传入：

```bash
bash scripts/train.sh configs/denoise_baseline.yaml "" --steps 200 --batch-size 1
```

> 上面第二个参数为空字符串表示不挂 profile；后续参数会作为 `denoise_baseline.py` 的 CLI 覆盖项。

### 4.2 检查点与日志保存位置

每次 `scripts/train.sh` 运行会在 `experiments/runs/<name>_<timestamp>/` 下写入：

- `config.yaml` / `config.final.yaml`：实际生效的配置副本
- `profile.yaml`：若使用了 profile
- `env.txt`：`scripts/check_env.py` 的输出
- `data.txt`：`scripts/check_data.py` 的输出
- `meta.txt`：命令行、git status、时间戳
- `train.log`：训练实时日志

模型 checkpoint 由配置 `paths.ckpt` 决定，默认形如：

```text
experiments/<exp_name>/<filename>.pkl
experiments/<exp_name>/<filename>.train.csv   # 同名 CSV 训练曲线
experiments/<exp_name>/last_run_summary.txt   # 最近一次训练摘要
```

按 `train.save_every` 周期覆盖保存；训练结束时会再保存一次最终 ckpt。

## 5. 推理 / 预测流程

### 5.1 命令示例

`scripts/predict.sh` 串联了三步：`predict` → `zip` → `validate-zip`：

```bash
bash scripts/predict.sh configs/denoise_baseline.yaml
# 或者带 profile（可控制 patch_size）
bash scripts/predict.sh configs/denoise_baseline.yaml configs/profiles/a6000.yaml
```

也可分步运行：

```bash
$PYTHON denoise_baseline.py --config configs/denoise_baseline.yaml --mode predict
$PYTHON denoise_baseline.py --config configs/denoise_baseline.yaml --mode zip
$PYTHON denoise_baseline.py --config configs/denoise_baseline.yaml --mode validate-zip
```

正式提交建议用统一推理 + 提交检查脚本：

```bash
$PYTHON scripts/unified_predict.py \
  --name <candidate> \
  --patch-size 8192 \
  --out-dir results/<candidate> \
  --zip result_<candidate>.zip
$PYTHON scripts/check_submission.py result_<candidate>.zip \
  --test-root dataset_test_noisy \
  --require-float32
```

### 5.2 输入输出格式

- **输入**：`<test_root>/shapenet/<synset>/<model>/noisy.npy`，`(N, 3) float32` 点云。
- **输出**：`<out_dir>/shapenet/<synset>/<model>/denoised.npy`，与输入同形状的去噪后点云。
- **提交 zip**：根目录是 `shapenet/<synset>/<model>/denoised.npy` 的相对路径，**不**包含外层目录。
  `validate_zip` 会校验路径层级与命名。

### 5.3 批量处理

`predict(args)` 会自动遍历 `<test_root>/shapenet/*/*/noisy.npy`：

- 命令行加 `--limit N` 或在 `predict.limit` 中设非零值，可只跑前 `N` 个样本，便于快速回归。
- 大点云使用 `predict_points_in_chunks(model, noisy_np, patch_size)` 做分块推理：
  - `patch_size` 来自 `predict.patch_size`，默认 `1000`，A6000 profile 提到 `8192`；
  - 分块互不重叠，**不**做 stitching，所以 `patch_size` 越大、单次显存占用越高、边界效应越小。
  - 显存吃紧时把 `patch_size` 调小即可。

## 6. 常见问题排查

### 6.1 CUDA 相关问题

- **`jt.flags.use_cuda` 报错或 fallback 到 CPU**：先确认 `nvidia-smi` 能正常显示 GPU；其次确认
  `cupy-cuda12x` 已安装且 import 成功（`scripts/install_deps.sh` 末尾会调用 `check_env.py` 自动检查）。
- **CUDA 版本不匹配**：本项目假设 CUDA 12.x。如果机器是 CUDA 11.x，需要手动安装与之匹配的
  `cupy-cuda11x` 轮子；CuPy 必须严格匹配运行时 CUDA。
- **`libcusolver.so.11` 找不到**：`STAASv0` 已显式避免使用 `jt.linalg.eigh`，因此即使缺失也不会
  阻止训练；如果是其它脚本触发，请安装与 GPU 驱动匹配的 cuSOLVER 库。
- **`gcc: error: unrecognized command line option`**：通常是 Jittor 选到了过新的 gcc。安装 gcc-10/g++-10
  并把 `CC` / `CXX` 指向它，或确保 `starter_code/.toolchain-gcc10/` 存在。
- **scalar 日志全是 `nan`**：`scalar()` 在缺 CuPy 时会返回 NaN。安装 `cupy-cuda12x` 即可。

### 6.2 内存 / 显存不足

- **GPU OOM（训练）**：依次降低 `train.batch_size`、`train.num_points`、`model.feat_dim/hidden`，或换更小的 profile。
- **GPU OOM（推理）**：把 `predict.patch_size` 调小（例如从 8192 降到 4096 / 1000）。
- **CPU 内存爆掉**：通常是 `cache_clean=true` 把全部 OBJ 载入内存所致；改回 `false`，或降低
  `train.limit` / `prefetch_queue_size`。
- **预取线程异常**：把 `train.prefetch_workers` 改为 `0` 退回到主线程加载，定位是否 IO/数据问题。

### 6.3 数据路径问题

- **`no obj files found from <list> under <root>`**：`ObjDenoiseDataset` 找不到任何样本。检查：
  - `dataset_train` 软链接是否生效（`ls -l dataset_train`）；
  - 训练列表里的相对路径是否能拼出真实 OBJ；
  - 如果列表条目缺少 `shapenet/` 前缀，仓库已自动兼容 `<data_root>/shapenet/<row>/...`。
- **`test_root not found` / `ckpt not found`**：`check_paths` 会在 train/predict 入口就抛错，
  按提示补齐路径或改用绝对路径。
- **提交 zip 校验失败**：路径必须是 `shapenet/<synset>/<model>/denoised.npy` 四层；
  推理直接由 `predict.sh` 完整跑完即可，不要手动拼装 zip。

### 6.4 Jittor 编译缓存

- **首次启动卡住或反复编译**：Jittor 会在 `JITTOR_HOME/.cache/jittor` 下编译算子。
  `scripts/env.sh` 默认把它指向仓库内 `.jittor_home`，请确保该目录可写、磁盘空间充足。
- **缓存损坏导致诡异错误**：可以安全地删除 `.jittor_home/.cache/jittor` 让 Jittor 重建：

  ```bash
  rm -rf .jittor_home/.cache/jittor
  ```

  下一次运行会重新编译，第一次会比较慢。
- **多用户共享机器**：可以把 `JITTOR_HOME` 指向用户私有目录，避免不同用户写同一缓存：

  ```bash
  export JITTOR_HOME=$HOME/.jittor_home_pcdn
  ```

- **gcc 切换后缓存仍报错**：换编译器后建议清空 `.jittor_home/.cache/jittor`，让 Jittor 重新生成与新工具链匹配的算子缓存。

---

如需更细的复现流程、候选登记或机器 profile 推荐，请参考：

- [`README_REPRODUCE.md`](../README_REPRODUCE.md)
- [`docs/REPOSITORY_STRUCTURE_CN.md`](REPOSITORY_STRUCTURE_CN.md)
- [`docs/GPU_PROFILES.md`](GPU_PROFILES.md)
- [`docs/CANDIDATE_REGISTRY.md`](CANDIDATE_REGISTRY.md)
- [`docs/API_REFERENCE.md`](API_REFERENCE.md)
