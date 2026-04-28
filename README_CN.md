# jittor-woshinailongduibu-pointcloud-denoise 中文说明

本仓库是 **计图 Jittor 点云降噪正式赛** 的代码仓库，包含由 **wupeiwei（小冷）本人提出** 的 **PW-SENEL / ST-AAS** 边缘保持点云降噪模块。

- 战队名：我是奶龙对不
- GitHub 账号：`wupeiwei`
- Gitee / GitLink 用户：`wpwwgw` / 小冷
- 框架：Jittor
- 任务：点云降噪 Point Cloud Denoising

## 1. 项目简介

本项目围绕正式赛点云降噪任务构建，目标是在保持点云几何结构、边缘和尖锐细节的同时，降低输入噪声。

整体思路采用残差预测：

```text
pred_clean = noisy + offset
```

也就是模型不直接生成完整干净点云，而是预测每个点需要移动的位移 `offset`。

## 2. 当前方法

### 2.1 普通残差降噪 baseline

基础模型使用官方 starter code 中的几何特征提取模块，结合 MLP offset head 输出点级位移。

特点：

- 结构简单，便于复现
- 适合作为后续 ablation 的基线
- 默认配置保持稳定，不和实验模块混在一起

### 2.2 PW-SENEL 模块

**PW-SENEL 是 wupeiwei（小冷）本人在本项目中提出的边缘保持点云降噪模块。**

PW-SENEL 全称：

```text
PeiWei Softmax Edge-aware Noise Elimination and Locking
```

中文名：

```text
沛慰软筛噪声-边缘锁定模块
```

核心思想：

- 使用 softmax / attention 风格的邻域加权来弱化疑似噪声点
- 使用边缘感知结构锁定来保护棱角、尖锐区域，避免过度平滑
- 在降噪时尽量保留局部几何细节

该模块目前作为可开关 ablation 实现，方便与 baseline 对比。

### 2.3 ST-AAS v0 模块

ST-AAS v0 全称：

```text
Structure Tensor-guided Adaptive Softmax
```

中文可理解为：

```text
结构张量引导的自适应 Softmax 降噪
```

**ST-AAS v0 是 wupeiwei（小冷）本人基于 PW-SENEL 思路进一步提出的轻量几何实现。**

核心流程：

1. 对每个点建立 KNN 邻域
2. 根据邻域密度自适应调整 softmax 温度 `tau_i`
3. 通过结构张量特征估计局部几何类型
   - linearity：线状结构
   - planarity：面状结构
   - scattering：散乱程度
4. 计算边缘置信度 `edge_conf`
5. 对普通区域做平滑，对边缘区域抑制过度平滑

当前边缘保护形式为：

```text
pred_i = p_i + (1 - edge_conf_i) * (smooth_i - p_i)
```

直观解释：

- 如果是平滑区域，可以多参考邻域平均结果
- 如果是边缘/尖锐区域，就少动一点，避免把边缘磨平

## 3. 目录结构

```text
.
├── denoise_baseline.py              # 正式赛主入口：训练 / 预测 / 打包
├── configs/
│   ├── denoise_baseline.yaml        # 普通 baseline
│   ├── denoise_pwsenel.yaml         # PW-SENEL 实验配置
│   ├── denoise_staas_v0.yaml        # ST-AAS v0 实验配置
│   └── profiles/                    # 不同机器的 profile
├── scripts/
│   ├── env.sh                       # 环境变量设置
│   ├── install_deps.sh              # 依赖安装
│   ├── train.sh                     # 训练封装脚本
│   ├── predict.sh                   # 预测封装脚本
│   ├── check_env.py                 # 环境检查
│   └── check_data.py                # 数据检查
├── starter_code/                    # 官方 starter code，尽量保持独立
├── docs/
│   ├── GPU_PROFILES.md              # 不同显卡配置说明
│   └── ST_AAS_CVMJ_PLAN.md          # ST-AAS 设计文档
├── README.md                        # 英文说明
├── README_CN.md                     # 中文说明
├── README_REPRODUCE.md              # 复现说明
├── OPEN_SOURCE.md                   # 开源说明
├── requirements.txt
└── .gitignore
```

## 4. 环境准备

推荐环境：

- Python 3.10-3.12
- Jittor >= 1.3.9
- CUDA GPU
- CUDA 12.x 环境建议安装 `cupy-cuda12x>=13,<14`

安装依赖：

```bash
cd jittor-woshinailongduibu-pointcloud-denoise
source scripts/env.sh
bash scripts/install_deps.sh
```

如果国内网络下载 CuPy 很慢，可以使用清华源：

```bash
"$PYTHON" -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "cupy-cuda12x>=13.0,<14.0"
```

检查环境：

```bash
source scripts/env.sh
"$PYTHON" scripts/check_env.py
```

## 5. 数据准备

正式数据集不要上传到 git。请在仓库根目录准备：

```text
dataset_train/
dataset_test_noisy/
```

可以是实体目录，也可以是软链接：

```bash
ln -s /path/to/pointcloud-denoise/dataset_train ./dataset_train
ln -s /path/to/pointcloud-denoise/dataset_test_noisy ./dataset_test_noisy
```

检查数据：

```bash
source scripts/env.sh
"$PYTHON" scripts/check_data.py --limit 5
```

## 6. 训练命令

普通 baseline：

```bash
bash scripts/train.sh configs/denoise_baseline.yaml configs/profiles/local_dev.yaml
```

PW-SENEL：

```bash
bash scripts/train.sh configs/denoise_pwsenel.yaml configs/profiles/local_dev.yaml
```

ST-AAS v0：

```bash
bash scripts/train.sh configs/denoise_staas_v0.yaml configs/profiles/local_dev.yaml
```

RTX 5060 Ti：

```bash
bash scripts/train.sh configs/denoise_staas_v0.yaml configs/profiles/rtx5060ti.yaml
```

RTX A6000：

```bash
bash scripts/train.sh configs/denoise_staas_v0.yaml configs/profiles/a6000.yaml
```

推荐流程：

```text
本机：代码编写、debug、小规模 smoke test
RTX 5060 Ti：中等规模实验、ablation 预筛
RTX A6000：大规模正式训练、最终实验
```

## 7. 预测与提交

训练完成后使用对应配置进行预测：

```bash
bash scripts/predict.sh configs/denoise_baseline.yaml configs/profiles/local_dev.yaml
```

也可以直接调用主入口：

```bash
source scripts/env.sh
"$PYTHON" denoise_baseline.py --config configs/denoise_baseline.yaml --mode predict
"$PYTHON" denoise_baseline.py --config configs/denoise_baseline.yaml --mode zip
"$PYTHON" denoise_baseline.py --config configs/denoise_baseline.yaml --mode validate-zip
```

## 8. 复现说明

更详细的复现流程请看：

- `README_REPRODUCE.md`
- `docs/GPU_PROFILES.md`
- `docs/ST_AAS_CVMJ_PLAN.md`

注意：运行 Jittor 相关命令前，建议先执行：

```bash
source scripts/env.sh
```

这样可以加载项目约定的 Python、编译器和 CUDA/Jittor 环境配置。

## 9. 不要上传的内容

`.gitignore` 已经排除了以下内容：

```text
dataset_train/
dataset_test_noisy/
starter_code/.venv/
experiments/
results/
*.pkl
*.zip
*.npy
*.npz
.env
密钥 / token
```

如果需要公开实验结果，建议单独整理表格、曲线图片或报告，不要直接上传完整训练产物和数据集。

## 10. 开源说明

官方要求仓库名格式：

```text
jittor-[战队名]-[项目名]
```

本仓库使用英文拼音形式：

```text
jittor-woshinailongduibu-pointcloud-denoise
```

对应中文战队名：

```text
我是奶龙对不
```

## 11. 许可证与引用

本项目采用 Apache License 2.0 开源，详见：

```text
LICENSE
NOTICE
```

PW-SENEL 与 ST-AAS 是 wupeiwei（小冷）本人在本项目中的原创竞赛贡献；“我是奶龙对不”仅作为参赛队伍名称。若复用、修改或二次开发相关模块，请保留署名信息，并引用本仓库。引用信息见：

```text
CITATION.cff
```

开源代码不等于放弃商业化权利。后续增强版、工程部署版、模型服务、咨询服务等仍可独立商业化。

## 12. 致谢

感谢：

- Jittor 框架
- 官方点云降噪 starter code
- EdgeConv / DGCNN 等点云几何学习方法
