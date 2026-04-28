# Open Source Guide / 开源说明

This document is prepared according to the official open-source requirement for the Jittor point cloud competition.

## 1. Official Requirement Summary

According to the official guide:

- Warm-up track: open-source on GitLink can receive souvenir rewards.
- A榜: contestants who promise to open-source code on GitHub and GitLink after B榜 closes are eligible to enter B榜.
- B榜: code must be open-sourced on GitHub and GitLink to receive prize money and attend the award ceremony.

Repository naming format:

```text
jittor-[team-name]-[project-name]
```

For this project, recommended name:

```text
jittor-woshinailongduibu-pointcloud-denoise
```

Known account information:

- GitHub: `wupeiwei`
- Gitee / GitLink nickname: 小冷

## 2. Recommended Remote Strategy

Because GitHub access may be unstable from the local network, use this practical strategy:

1. Use Gitee/GitLink as the daily stable remote.
2. Add GitHub as the official mirror/backup remote.
3. Keep both remotes synchronized before official open-source checkpoints.

Example remote layout:

```bash
git remote add origin https://gitee.com/<username>/jittor-woshinailongduibu-pointcloud-denoise.git
git remote add github https://github.com/wupeiwei/jittor-woshinailongduibu-pointcloud-denoise.git
```

Daily push:

```bash
git push origin main
```

Official sync:

```bash
git push origin main
git push github main
```

If GitLink is separate from Gitee in the official portal, create/import the same repository there and add another remote, for example:

```bash
git remote add gitlink <gitlink-repository-url>
git push gitlink main
```

## 3. Files That Should Be Open-Sourced

Commit these:

```text
denoise_baseline.py
configs/
scripts/
docs/
starter_code/ core source/config files
README.md
README_REPRODUCE.md
OPEN_SOURCE.md
requirements.txt
.gitignore
LICENSE
```

## 4. Files That Must Not Be Committed

Do not commit:

```text
dataset_train/
dataset_test_noisy/
data/
starter_code/.venv/
experiments/
results/
*.pkl
*.zip
*.npy
*.npz
*.log
.env
private tokens / keys
```

The `.gitignore` file has been prepared for this.

## 5. First-Time Git Setup

From the project root:

```bash
cd /home/sallen/jittor-pointcloud-denoise
git init
git branch -M main
```

Check what will be committed:

```bash
git status --short
git check-ignore -v dataset_train dataset_test_noisy starter_code/.venv experiments results || true
```

Add files:

```bash
git add .
git status --short
```

If the status looks clean and contains no dataset/checkpoint/venv files:

```bash
git commit -m "init formal point cloud denoising project"
```

## 6. Push to Gitee/GitLink First

```bash
git remote add origin https://gitee.com/<username>/jittor-woshinailongduibu-pointcloud-denoise.git
git push -u origin main
```

Replace `<username>` with the actual Gitee account path. The nickname 小冷 may not be the URL username.

## 7. Add GitHub Mirror Later

```bash
git remote add github https://github.com/wupeiwei/jittor-woshinailongduibu-pointcloud-denoise.git
git push -u github main
```

If HTTPS login is unstable or asks for a password, use a GitHub personal access token or SSH key.

## 8. A6000 Training Machine Workflow

On the A6000 machine:

```bash
git clone https://gitee.com/<username>/jittor-woshinailongduibu-pointcloud-denoise.git
cd jittor-woshinailongduibu-pointcloud-denoise
source scripts/env.sh
bash scripts/install_deps.sh
"$PYTHON" scripts/check_env.py
"$PYTHON" scripts/check_data.py --limit 5
```

Then train:

```bash
bash scripts/train.sh configs/denoise_staas_v0.yaml configs/profiles/a6000.yaml
```

Datasets should be placed separately and linked into the repo:

```bash
ln -s /path/to/pointcloud-denoise/dataset_train ./dataset_train
ln -s /path/to/pointcloud-denoise/dataset_test_noisy ./dataset_test_noisy
```

## 9. Before Public Release Checklist

Before making the repository public or submitting the open-source link:

```bash
git status --short
git log --oneline -5
git ls-files | grep -E 'dataset|\.pkl$|\.zip$|\.npy$|\.npz$|\.env|token|key|\.venv' || true
```

Also check:

- README has correct training commands.
- Dataset path instructions are clear.
- No private credentials are committed.
- Official starter code attribution is kept.
- Results/weights are only uploaded if allowed by the competition.

## 10. Suggested Commit Milestones

```text
init formal point cloud denoising project
add reproducible configs and gpu profiles
add pw-senel ablation module
add st-aas v0 module and docs
fix jittor cuda/cupy dependency notes
prepare open-source documentation
```
