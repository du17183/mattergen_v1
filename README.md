# Q3 E3-PCR 冻结 64-seed 验证

> 当前分支：`feature/q3-e3-pcr-frozen64`

本分支在快速筛选后冻结 Q3 配置，使用 64 个新 seeds 验证效果、质量安全性，并检查学习门控的机制贡献。

## 冻结结论

```text
Q3_FROZEN_64_GO=True
EFFECT_GO=True
QUALITY_GO=True
SAFETY_GO=True
MECHANISM_SUPPORTED=False
SEEDS=32000–32063
N=64
```

| 指标 | C0 | Q3 | 变化 |
|---|---:|---:|---:|
| 预松弛最大力 | 0.397573 | 0.264137 | **-33.5625%** |
| Relaxation RMSD | — | — | -2.376% |
| E-hull / Stable / NUS | — | — | 不变 |
| Novel / Unique | — | — | 不变 |

```text
bootstrap 95% CI=[-0.261324, -0.048601]
Wilcoxon p=1.4196e-8
Win/Loss=52/12
```

## 机制消融

Always-on 的最大力改善为 `-36.14%`、harm rate 为 `9.375%`；学习门控的 harm rate 为 `18.75%`。因此 64-seed 数据支持 Q3 整体有效，但不支持“学习门控优于 always-on”的机制主张。

## 工作过程

1. 冻结筛选阶段选择的 Q3 配置。
2. 使用 64 个未参与筛选的新 seeds。
3. 完成 C0/Q3 配对生成和 MatterSim relaxation。
4. 评估效果、质量、安全与门控机制四个 gate。
5. 仅将整体模块推进到独立 256-seed 正式验证。

## 证据

- [冻结 64-seed 报告](reports/q3_e3_pcr/frozen64/final_report.md)
- 正式结果：`feature/q3-e3-pcr-formal256`
- [论文归档分支](https://github.com/du17183/mattergen_v1/tree/archive/thesis-analysis-package-v1)
