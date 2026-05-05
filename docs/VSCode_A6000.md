# VSCode on A6000

Use this project as a normal VSCode / VSCode Remote-SSH workspace.

## 1. Open workspace

From VSCode:

1. Connect to the A6000 server with Remote-SSH.
2. Open this repository folder.
3. Optional: open `jittor-pointcloud-denoise.code-workspace`.

## 2. Python environment

The repository tasks assume the project environment script is used:

```bash
source scripts/env.sh
```

By default `scripts/env.sh` looks for:

```text
starter_code/.venv/bin/python
```

If the server uses another virtualenv path, set `VENV_PATH` before running tasks:

```bash
export VENV_PATH=/path/to/venv
source scripts/env.sh
```

You can also override Python directly:

```bash
PYTHON=/path/to/python bash scripts/train.sh configs/denoise_baseline.yaml configs/profiles/a6000.yaml
```

## 3. Recommended VSCode tasks

Press `Ctrl+Shift+P` → `Tasks: Run Task`, then run:

1. `check env`
2. `check data a6000`
3. `train baseline a6000` or `train adaptive_clip a6000`
4. `evaluate adaptive_clip low/mid/high`

## 4. Notes

- Do not edit official `starter_code` unless necessary.
- Keep datasets/checkpoints/results out of git.
- If Jittor compiler errors mention GCC/CUDA, fix environment first before changing model code.
