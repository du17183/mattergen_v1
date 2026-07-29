# CrystalREPA 无条件 MatterGen 复现

> 当前分支：`feature/crystalrepa-repro`

本分支隔离验证 CrystalREPA-like representation alignment 能否在无条件 MP-20 MatterGen 上复现稳定性改善，用于判断此前条件 FN-PRA 失败究竟来自实现还是任务设置。

## 最终结论

```text
REPA_REPRO_ENGINEERING_GO=False
REPA_REPRO_SCIENTIFIC_GO=False
REPA_BASE_REPRODUCED=False
REPA_REPRO_NO_GO=True
TRAINING_STEPS=10000
EVAL_SEEDS=17000–17063
N=64
```

| 指标 | R1 相对 U0 |
|---|---:|
| Composition validity | -3.125 pp |
| 平均 E-hull | +0.094236 eV/atom |
| Metastable | -6.25 pp |
| Stable | 0 pp |
| Relaxation RMSD | +0.03329 |

## 工作过程

1. 切换到官方无条件 MP-20 checkpoint，关闭 Adaptive CFG。
2. 核对 Teacher cache、原子映射和 EA-NCE mask。
3. 将对齐点调整到中间 GemNet block。
4. 使用 CHGNet Teacher 训练至 10,000 steps 并保存验证最佳点。
5. 完成 64-seed U0/R1 严格配对生成和 MatterSim relaxation。

## 如何理解

该复现没有得到论文方向上的改善。可能差异包括 Teacher、checkpoint、训练规模或实验细节，但在本项目可验证配置下，继续把 REPA 叠加到条件 FN-PRA 没有充分依据。

## 证据

- [最终报告](research/crystalrepa_repro/artifacts/reports/crystalrepa_repro_final_report.md)
- [论文归档分支](https://github.com/du17183/mattergen_v1/tree/archive/thesis-analysis-package-v1)

```text
STABILITY_SOURCE=MatterSim-5M surrogate
DFT_VERIFIED=False
```
