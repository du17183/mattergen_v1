# 创新点二正式验证：Q3 E3-PCR

> 当前分支：`feature/q3-e3-pcr-formal256`

本分支冻结第二创新点 **Q3 E3-PCR Learned-Gated Post-generation Refinement** 的独立 256-seed 正式结果。

## 冻结结论

```text
FINAL_STATE=E3_G_FORMAL_CONFIRMED
FROZEN_COMMIT=0275cbf08ed3c6321cea7d06f7a3a8edb83b7483
FORMAL_SEEDS=40000–40255
N=256
```

| 指标 | E3-G 相对 C0 |
|---|---:|
| 预松弛最大力 | **-23.28%** |
| Harm rate | 18.359% |
| 原子种类 | 不修改 |
| MatterGen 主干 | 不重新训练 |
| 采样轨迹 | 不修改 |

## 方法

Q3 在生成结束后执行受控的等变局部修正。一个 129 参数门控器根据 14 个结构和力学特征判断是否接受 E3-PCR 候选；trust region 和 fallback 负责限制不安全更新。

## 工作过程

1. 从六个后生成质量模块中快速筛选候选。
2. 在新 32-seed 池中确认 Q3 的力改善和质量安全性。
3. 进行 64-seed 冻结验证，确认效果，但发现“学习门控优于 always-on”的机制证据不足。
4. 使用完全独立的 256 seeds 做最终检验。
5. 冻结模型、阈值、统计口径和正式 commit。

## 如何理解结果

- 正向结论是降低预松弛最大力，而不是证明 DFT 稳定性提升。
- 该模块不改变生成采样轨迹，因此可以与 Adaptive CFG 串联。
- 论文应如实说明 64-seed 阶段的门控机制消融不充分；正式结论以独立 256-seed 的整体模块效果为准。

## 证据入口

正式报告和统计文件位于：

- [`reports/q3_e3_pcr/formal256/`](reports/q3_e3_pcr/formal256/)
- [论文归档分支](https://github.com/du17183/mattergen_v1/tree/archive/thesis-analysis-package-v1)

## 科学边界

```text
STABILITY_SOURCE=MatterSim-5M surrogate
DFT_VERIFIED=False
PROPERTY_TARGET_VERIFIED=False
```
