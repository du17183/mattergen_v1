# Budget-Aware Corrector Gating

> 当前分支：`feature/budget-aware-corrector-gating`

本分支开发预算感知的收敛引导 Corrector 调度器，目标是直接减少完整物理模型 forward，而不是只减少 CFG 联合 batch 中的 unconditional 样本。

## 最终结论

```text
BUDGET_AWARE_GATING_COMPLETED=True
FINAL_SCIENTIFIC_DECISION=NO_GO
DEVELOPMENT_SEEDS=14000–14031
N=32
```

| 配置 | 单任务加速 | 物理 forward | Stable | NUS | E-hull |
|---|---:|---:|---:|---:|---:|
| G1 | 1.183× | -12.63% | -3.125 pp | -6.25 pp | -0.00384 |
| G2 | 1.234× | -20.83% | -3.125 pp | 0 pp | +0.02229 |

没有配置同时通过冻结的速度、计算量和质量门槛，因此没有进入 64-seed 验证。

## 工作过程

1. 基于 residual、结构更新幅度和稳定步数实现门控。
2. 加入计算预算、Atomic veto、校准、fallback 和统计计数。
3. 完成 52 项专项测试及 8-seed smoke。
4. 完成 32-seed A0/G1/G2 配对生成与 MatterSim 评价。
5. 根据预先冻结门槛判定 No-Go 并停止扩展。

## 重要发现

Corrector 跳过能够真实减少物理 forward 并带来 13%–22% 吞吐提升，但质量损失随跳过率增加。此前更激进的 G3 在 256 seeds 上达到 1.506×，同时 Stable、NUS 各下降约 9–10 pp，进一步确认速度—质量矛盾。

## 证据

- [最终报告](research/budget_aware_gating/reports/final/budget_aware_final_report.md)
- [论文归档分支](https://github.com/du17183/mattergen_v1/tree/archive/thesis-analysis-package-v1)

本分支应作为负面消融和工程经验，不应作为最终第二创新点。

## 算法实现

调度器为每个 timestep 保留完整 Predictor，并根据收敛状态决定是否执行 Corrector：

```text
warmup full Predictor + Corrector
→ 读取 residual EMA、更新幅度、预算使用率和 atomic risk
→ stable 且预算允许：跳过本步 Corrector
→ 周期校准或异常：恢复完整 Corrector
→ Atomic veto / calibration / fallback 保证安全
```

与 unconditional reuse 不同，跳过 Corrector 会省掉一次完整模型 forward，所以能够转化为真实 wall-time 收益。

## 实现位置

| 文件 | 内容 |
|---|---|
| [`corrector_gating.py`](mattergen/diffusion/sampling/corrector_gating.py) | 收敛判定、预算、校准、veto 和统计 |
| [`pc_sampler.py`](mattergen/diffusion/sampling/pc_sampler.py) | 在采样循环接入 gate |
| [`G1.json`](research/budget_aware_gating/configs/G1.json) | 保守候选 |
| [`G2.json`](research/budget_aware_gating/configs/G2.json) | 中等候选 |
| [`test_budget_aware_corrector_gating.py`](mattergen/diffusion/tests/test_budget_aware_corrector_gating.py) | 预算、fallback、计数和边界测试 |

## 实验数据

- 8-seed smoke 与确定性结果：[eight_seed_go_no_go.md](research/budget_aware_gating/reports/eight_seed_go_no_go.md)
- 32-seed A0/G1/G2 逐样本记录：[paired_seed_records.csv](research/budget_aware_gating/reports/final/paired_seed_records.csv)
- 速度—质量 Pareto：[pareto_comparison.csv](research/budget_aware_gating/reports/final/pareto_comparison.csv)
- 候选判定：[candidate_decisions.json](research/budget_aware_gating/reports/final/candidate_decisions.json)
- 复现命令：[reproduction_commands.md](research/budget_aware_gating/reports/final/reproduction_commands.md)

## 运行与验证

```bash
bash research/budget_aware_gating/scripts/status_budget_aware.sh
python -m pytest mattergen/diffusion/tests/test_budget_aware_corrector_gating.py -q
```

完整 runner 为 `research/budget_aware_gating/scripts/run_budget_aware.sh`。本分支保留 No-Go 是因为没有候选同时满足速度与质量门槛，不是因为跳过逻辑没有运行。
