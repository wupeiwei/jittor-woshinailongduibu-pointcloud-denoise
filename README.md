# jittor-woshinailongduibu-pointcloud-denoise

Jittor implementation for point cloud denoising in the Jittor point cloud competition formal track.

This repository contains a reproducible denoising baseline and the proposed **ST-AAS / PW-SENEL** module for edge-preserving point cloud denoising.

- Team name: 我是奶龙对不
- Author / GitHub account: wupeiwei
- GitHub account: `wupeiwei`
- Gitee / GitLink nickname: 小冷
- Framework: [Jittor](https://github.com/Jittor/jittor)
- Task: point cloud denoising

## 1. Method Overview

The project starts from a residual point-wise denoising baseline:

```text
pred_clean = noisy + offset
```

The baseline uses official starter-code geometric feature extraction modules, while keeping custom competition code independent from official files.

Current optional modules:

1. **Baseline residual denoiser**
   - EdgeConv-style feature extraction
   - MLP offset head
   - Chamfer-style auxiliary loss support

2. **PW-SENEL**
   - PeiWei Softmax Edge-aware Noise Elimination and Locking
   - Original idea: softmax-based noise suppression + max-pooling edge locking
   - Implemented as a switchable ablation module

3. **ST-AAS v0**
   - Structure Tensor-guided Adaptive Softmax
   - Single KNN neighborhood
   - Density-adaptive softmax temperature
   - Structure-tensor descriptors: linearity, planarity, scattering
   - Edge-aware smoothing suppression:

```text
pred_i = p_i + (1 - edge_conf_i) * (smooth_i - p_i)
```

This design targets lightweight, interpretable, edge-preserving denoising with low training overhead and good reproducibility.

## 2. Repository Structure

```text
.
├── denoise_baseline.py              # main training / prediction / zip entry
├── configs/
│   ├── denoise_baseline.yaml        # baseline config
│   ├── denoise_pwsenel.yaml         # PW-SENEL ablation config
│   ├── denoise_staas_v0.yaml        # ST-AAS v0 ablation config
│   └── profiles/                    # machine-specific profiles
├── scripts/
│   ├── env.sh                       # environment setup
│   ├── install_deps.sh              # dependency installation
│   ├── train.sh                     # reproducible training wrapper
│   ├── predict.sh                   # prediction wrapper
│   ├── check_env.py                 # environment check
│   └── check_data.py                # dataset check
├── starter_code/                    # official starter code, kept mostly unchanged
├── docs/
│   ├── GPU_PROFILES.md
│   └── ST_AAS_CVMJ_PLAN.md
├── README_REPRODUCE.md              # detailed reproduction notes
├── requirements.txt
└── .gitignore
```

Large files are intentionally excluded from git:

- datasets
- virtual environments
- checkpoints
- experiment logs
- prediction zip files

## 3. Environment

Recommended:

- Python 3.10-3.12
- Jittor >= 1.3.9
- CUDA-enabled GPU for training
- CUDA 12.x machines may need `cupy-cuda12x>=13,<14`

Install dependencies:

```bash
cd jittor-woshinailongduibu-pointcloud-denoise
source scripts/env.sh
bash scripts/install_deps.sh
```

If CuPy installation is slow in mainland China, use a mirror:

```bash
"$PYTHON" -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "cupy-cuda12x>=13.0,<14.0"
```

Check environment:

```bash
source scripts/env.sh
"$PYTHON" scripts/check_env.py
```

## 4. Data Preparation

Prepare the official formal-track dataset outside git, then create directories or symbolic links in the repository root:

```text
dataset_train/
dataset_test_noisy/
```

Example:

```bash
ln -s /path/to/pointcloud-denoise/dataset_train ./dataset_train
ln -s /path/to/pointcloud-denoise/dataset_test_noisy ./dataset_test_noisy
```

Check data:

```bash
source scripts/env.sh
"$PYTHON" scripts/check_data.py --limit 5
```

## 5. Training

Baseline:

```bash
bash scripts/train.sh configs/denoise_baseline.yaml configs/profiles/local_dev.yaml
```

PW-SENEL ablation:

```bash
bash scripts/train.sh configs/denoise_pwsenel.yaml configs/profiles/local_dev.yaml
```

ST-AAS v0 ablation:

```bash
bash scripts/train.sh configs/denoise_staas_v0.yaml configs/profiles/local_dev.yaml
```

For RTX 5060 Ti:

```bash
bash scripts/train.sh configs/denoise_staas_v0.yaml configs/profiles/rtx5060ti.yaml
```

For RTX A6000:

```bash
bash scripts/train.sh configs/denoise_staas_v0.yaml configs/profiles/a6000.yaml
```

Recommended workflow:

```text
local machine: smoke test and debugging
RTX 5060 Ti: medium experiments and ablations
RTX A6000: large formal training runs
```

## 6. Prediction and Submission

After training, run prediction with the selected checkpoint:

```bash
bash scripts/predict.sh configs/denoise_baseline.yaml configs/profiles/local_dev.yaml
```

The script writes prediction results and submission zip according to the config paths.

You can also use the direct entry:

```bash
source scripts/env.sh
"$PYTHON" denoise_baseline.py --config configs/denoise_baseline.yaml --mode predict
"$PYTHON" denoise_baseline.py --config configs/denoise_baseline.yaml --mode zip
"$PYTHON" denoise_baseline.py --config configs/denoise_baseline.yaml --mode validate-zip
```

## 7. Reproducibility

Each wrapper run saves run metadata, config/profile information, and logs under the experiment directory. See:

- `README_REPRODUCE.md`
- `docs/GPU_PROFILES.md`

Important environment convention:

```bash
source scripts/env.sh
```

This sets compiler and Python compatibility options for Jittor across local machines, RTX 5060 Ti, and RTX A6000 environments.

## 8. Open Source Notes

This repository is prepared for the official open-source requirement:

- A榜: code should be committed before entering B榜 qualification as required by the organizer.
- B榜: code should be open-sourced on GitHub and GitLink/Gitee according to official rules.

Suggested repository name:

```text
jittor-woshinailongduibu-pointcloud-denoise
```

Do not commit:

- official dataset files
- trained weights / checkpoints unless explicitly allowed
- virtual environments
- private tokens or credentials
- large result archives

## 9. License

MIT License. See `LICENSE`.

## 10. Acknowledgments

- Jittor framework
- Official point cloud denoising starter code
- EdgeConv / dynamic graph point cloud learning ideas
