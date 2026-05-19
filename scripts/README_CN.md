# scripts 目录说明

这个目录放工程脚本。当前先做轻量分组说明，不大规模移动主入口，避免破坏训练/预测/评估链路。

## 核心入口：先保留在 scripts 根目录

- `env.sh`：项目环境入口，训练/预测脚本会 source 它。
- `install_deps.sh`：安装项目依赖。
- `train.sh`：推荐训练入口。
- `predict.sh`：基础预测入口。
- `unified_predict.py`：统一推理/路由/提交输出入口。
- `candidate_registry.py`：候选提交登记入口。
- `check_env.py` / `check_data.py` / `check_submission.py`：环境、数据、提交包检查。
- `collect_runs.py`：汇总实验 run 信息。

## 评估 / 路由

- `evaluate_cd.py`：本地 synthetic CD 评估。
- `evaluate_noise_estimator.py`：噪声/roughness 估计器评估。
- `evaluate_router.py`：router 结果汇总分析。
- `predict_router.py`：旧 router 推理入口。
- `run_official_eval.py`：官方 evaluator 包装。

## A6000 专用脚本

文件名以 `a6000_` 开头，或 `activate_a6000_workspace.sh`。这些脚本服务远端 A6000 工作流，暂时不移动，避免内部 `source scripts/...` 路径失效。

## 分析 / 原型 / 诊断

- `prototype_move_scale_gate.py`：StraightPCF-style bounded move-scale gate 的 Jittor 算子兼容原型。
- `calibrate_move_scale_refs.py`：move-scale gate 的 roughness / KNN spacing 校准。
- `inspect_official_vm_patch_coverage.py`：官方 VM patch 覆盖诊断。
- `summarize_router_stress.py`：router stress 输出汇总。

## 官方 VM 包装

- `package_official_vm_outputs.py`：整理官方 VM 输出为正式提交结构。
- `check_official_quick_outputs.sh`：检查官方 quick outputs。

## 环境辅助

- `gpu_profile.py`：根据 GPU 给 profile 建议。
- `freeze_env.sh`：冻结环境信息。
- `make_wheelhouse.sh`：准备离线依赖 wheelhouse。

## smoke / legacy

- `smoke/run_config_smoke.sh`
- `smoke/run_denoise_predict_smoke.sh`
- `smoke/run_denoise_smoke.sh`
- `smoke/run_train_baseline.sh`

这些原来在项目根目录，现收进 `scripts/smoke/`，用于快速冒烟或旧流程参考。移动后已修正 `PROJECT_ROOT` 计算。
