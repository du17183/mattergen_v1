# Corrector residual distillation v1

这是第二创新的第一轮可复现实验：保留原始 Corrector，只用 2,661 参数的三字段轻量 Adapter 近似 Corrector 后、Predictor 前的第二次 MatterGen score；风险高、校准缺失或数值非有限时回退到原始 MatterGen。

本分支没有修改冻结的 Adaptive CFG 实现或其 formal256 结果。新功能默认关闭，只有显式设置 `sampler_partial.corrector_residual_adapter.enabled=true` 才启用。

## 第一轮结论

32 个独立 Stage-C seeds 中，`Adapter+Fallback@50` 是最值得复验的候选：实际 coverage 53.99%、MatterGen forward 减少 27.00%、配对 speedup 1.267×。其 E-hull/Stable/NUS 的描述性结果优于 C0，但 RMSD 和 pre-relaxation force 变差，因此尚不能宣称质量不下降，也不建议直接进入 formal256。完整正负结果、10 方法表格和下一步决策见 `final_report.md`。

## 固定实验设计

- 任务：`dft_mag_density=0.1`
- 采样：`N=1000`、`n_steps_corrector=1`、batch size 1、CFG scale 2.0
- deterministic，关闭 trajectory 记录
- baseline recovery: seeds 60000–60007
- teacher train: 61000–61015
- teacher validation: 62000–62007
- independent Stage-C test: 63000–63031
- coverage 仅由 validation 校准为 25/50/75/90%，不查看 Stage-C 质量结果调阈值

历史正式 seed 范围见 `config/experiment.json`，所有 train/validation/test 与历史范围均无交集。

## 方法

- C0：原始 constant CFG Predictor-Corrector
- A0：冻结的 Adaptive CFG
- Skip：移除 Corrector 的旧式跳过基线
- Reuse：保留 Corrector，但 Predictor 直接复用 Corrector 前 score
- Adapter：保留 Corrector，always-on residual Adapter
- Adapter+Fallback@25/50/75/90：按 validation 风险分位点回退 exact score
- A0+Adapter+Fallback@75：第一创新与第二创新组合

纯 C0 timing 单独运行，teacher recorder 的 I/O 时间不进入 speedup。所有 speedup 使用同 seed 的 C0 配对比值。

## 复现入口

```bash
cd /mnt/lis-wam-data/dxl/mattergen_v1
source experiments/corrector_residual_distillation_v1/activate.sh
PYTHONDONTWRITEBYTECODE=1 python -m pytest -s \
  mattergen/diffusion/tests/test_residual_distillation.py \
  mattergen/diffusion/tests/test_guidance_schedule.py -q

PYTHONDONTWRITEBYTECODE=1 \
  python research/corrector_distillation/audit_results.py
```

`-s` 用于规避本服务器项目盘禁止 pytest 临时捕获文件 unlink 的限制。主要脚本位于 `research/corrector_distillation/`；最终训练参数见 `config/final_adapter_training.json`。大型权重、teacher tensor、生成结构、日志和松弛轨迹只保存在项目目录，不进入 Git。
