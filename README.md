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
