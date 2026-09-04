# Corrector residual distillation v1

这是第二创新的第一轮可复现实验：保留原始 Corrector，只用 2,661 参数的三字段轻量 Adapter 近似 Corrector 后、Predictor 前的第二次 MatterGen score；风险高、校准缺失或数值非有限时回退到原始 MatterGen。

本分支没有修改冻结的 Adaptive CFG 实现或其 formal256 结果。新功能默认关闭，只有显式设置 `sampler_partial.corrector_residual_adapter.enabled=true` 才启用。

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
python -m pytest mattergen/diffusion/tests/test_residual_distillation.py \
  mattergen/diffusion/tests/test_guidance_schedule.py -q
```

主要脚本位于 `research/corrector_distillation/`；完整命令、结果表和结论见 `final_report.md`。大型权重、teacher tensor、生成结构和松弛轨迹只保存在项目目录，不进入 Git。
