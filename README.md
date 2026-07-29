# A0 + E3-G 256-seed 源数据审计

> 当前分支：`feature/a0-e3g-formal256`

本分支原计划进行 A0 + E3-G 的 256-seed 正式评估，但审计发现候选源数据包含 gate 训练 seeds，因此停止效果估计并保留审计证据。

## 冻结结论

```text
STATUS=SOURCE_DATA_INCOMPLETE — NO_EFFECT_ESTIMATE_PRODUCED
FROZEN_COMMIT=c1df24a8e5d118dcc99d7fb65b34e7f53be28969
CANDIDATE_SEEDS=20000–20255
TRAINING_OVERLAP=20000–20063
```

这个状态不是“方法 No-Go”，而是“数据源不满足独立验证条件”。本分支没有产生可以用于论文主结论的效果估计。

## 工作过程

1. 冻结候选 256-seed 输入与预期评估流程。
2. 追踪 E3-G gate 的训练 seed 清单。
3. 检测到 64 个 seeds 与候选正式集合重叠。
4. 停止合并统计，避免把训练泄漏写成独立验证。
5. 将后续工作拆为泄漏诊断和全新独立验证。

## 后续去向

- 泄漏程度分析：`experiment/a0-e3g-leakage-diagnostic256`
- 全新独立 64-seed 复现：`feature/a0-e3g-independent64`
- E3-G 自身独立 256-seed 正式结果：`feature/q3-e3-pcr-formal256`

## 结论用途

本分支适合展示实验治理过程：为什么发现问题、为什么停止、如何重新设计独立验证。不得将“未估计”改写成正向或负向科学结论。

完整项目脉络见[论文归档分支](https://github.com/du17183/mattergen_v1/tree/archive/thesis-analysis-package-v1)。
