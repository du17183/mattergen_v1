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

## 审计是如何实现的

[`research/a0_e3g_formal256.py`](research/a0_e3g_formal256.py) 的职责不是计算方法效果，而是：

1. 固定候选 source、branch、commit 和 seed 范围。
2. 读取 gate 训练 seed 清单。
3. 逐 seed 检查复用结构、哈希和指标覆盖。
4. 一旦发现训练重叠，就将状态写为 source-incomplete。
5. 停止正式 effect estimate，输出可恢复的 audit/report。

## 代码与证据

| 文件 | 内容 |
|---|---|
| [`research/a0_e3g_formal256.py`](research/a0_e3g_formal256.py) | `audit` 与 `status` 命令 |
| [`tests/test_a0_e3g_formal256.py`](tests/test_a0_e3g_formal256.py) | 数据源、seed 交集和禁止错误结论的测试 |
| [`reports/a0_e3g_formal256/reuse_audit.csv`](reports/a0_e3g_formal256/reuse_audit.csv) | 逐 seed 复用审计 |
| [`reports/a0_e3g_formal256/frozen_manifest.md`](reports/a0_e3g_formal256/frozen_manifest.md) | 冻结输入与范围 |
| [`reports/a0_e3g_formal256/final_report.md`](reports/a0_e3g_formal256/final_report.md) | 停止原因和后续建议 |

## 复核命令

```bash
python -m pytest tests/test_a0_e3g_formal256.py -q
python -m research.a0_e3g_formal256 status
python -m research.a0_e3g_formal256 audit
```

该分支有意不提供“效果对比表”：在 source contract 不成立时继续计算总体效果，会制造不可接受的训练—测试泄漏。
