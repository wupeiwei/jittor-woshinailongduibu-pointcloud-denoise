# Repository Structure and Reproduction Boundary

This document separates the files required for reproducing the code from files that only describe the author's own experiment workflow or machine-specific settings.

## 1. Reproduction-essential files

These files are needed to understand and run the main formal-track denoising code:

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

The optional method switches are controlled by config files:

- `configs/denoise_baseline.yaml`: residual denoising baseline
- `configs/denoise_pwsenel.yaml`: PW-SENEL ablation
- `configs/denoise_staas_v0.yaml`: ST-AAS v0 ablation

## 2. Required external files not stored in git

The official dataset is required for actual training and prediction, but it must not be committed to git:

```text
dataset_train/
dataset_test_noisy/
```

They may be real directories or symbolic links to directories outside the repository.

## 3. Official starter code

```text
starter_code/
```

This directory contains the official starter code and supporting modules. It is kept mostly independent from the author's formal-track implementation. The custom competition entry is `denoise_baseline.py`.

## 4. Author machine profiles

```text
configs/profiles/local_dev.yaml
configs/profiles/rtx5060ti.yaml
configs/profiles/a6000.yaml
```

These files are not part of the core algorithm. They document and support the author's own hardware workflow:

```text
local_dev: local debugging / smoke tests / log analysis
rtx5060ti: first complete runs and medium-scale ablation screening
a6000: larger formal training runs
```

Other users do not need identical hardware. They can run without a profile or create a new profile for their own GPU/CPU/memory budget.

## 5. Smoke-test and author convenience scripts

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

These scripts are useful for debugging, environment recording, offline dependency preparation, or the author's local workflow. They are not required to understand the core method.

## 6. Documentation and project metadata

```text
docs/GPU_PROFILES.md
docs/ST_AAS_CVMJ_PLAN.md
OPEN_SOURCE.md
LICENSE
NOTICE
CITATION.cff
```

- `docs/GPU_PROFILES.md`: explains the profile mechanism and author hardware examples.
- `docs/ST_AAS_CVMJ_PLAN.md`: design notes for future ST-AAS development.
- `LICENSE`, `NOTICE`, `CITATION.cff`: license, attribution, and citation metadata.

## 7. Files intentionally excluded from git

The following are not reproduction source files and should not be uploaded:

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

They are local data, checkpoints, logs, virtual environments, prediction outputs, or credentials.
