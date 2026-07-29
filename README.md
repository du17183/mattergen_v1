# 后生成质量模块快速筛选

> 当前分支：`feature/postgen-quality-modules-fastgate`

本分支从六个不重新训练 MatterGen 主干、不修改采样轨迹的后生成模块中筛选第二创新点候选，最终选择 Q3 E3-PCR。

## 最终结论

```text
Q3_E3_PCR_FINAL_GO=True
SCREENING_STAGE_COMPLETED=True
```

## 候选筛选

| 候选 | 主要结果 | 判定 |
|---|---|---|
| Q1 | Novel -14.86 pp | No-Go |
| Q2 | Novel -30.25 pp，Unique -8.45 pp | No-Go |
| Q4 | Novel -17.08 pp | No-Go |
| Q6 | 新 32-pool Novel -12.50 pp | No-Go |
| Q5 | Novel -15.63 pp | No-Go |
| Q3 | 新 32-seed 最大力 -20.454%，离散质量不变 | **GO** |

Q3 配对统计：

```text
bootstrap 95% CI=[-0.104730, -0.022368]
Wilcoxon p=0.000328
Win/Tie/Loss=16/12/4
```

## Q3 结构

- 等变 E3-PCR 后生成候选修正。
- 129 参数学习门控器。
- 14 个结构、力学和风险特征。
- trust region、质量约束和 fallback。
- 原子种类与 MatterGen 采样轨迹保持不变。

## 工作过程

1. 为六个候选定义共同的质量安全门槛。
2. 先复用已有结构做离线快速筛选。
3. 对可能受旧数据影响的候选启用全新 32-seed 池。
4. 独立计算 MatterSim relaxation 和配对统计。
5. 只冻结一个候选 Q3，转入 64/256-seed 验证。

## 证据与后续

- [筛选最终报告](reports/postgen_fastgate/final_report.md)
- 64-seed：`feature/q3-e3-pcr-frozen64`
- 256-seed：`feature/q3-e3-pcr-formal256`
- [论文归档分支](https://github.com/du17183/mattergen_v1/tree/archive/thesis-analysis-package-v1)
