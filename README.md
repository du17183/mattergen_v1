# SPG / Native Batch / BF16 GPU Fast-Gate

> 当前分支：`feature/spg-mattergen-fastgate`

本分支对 GPU 推理加速候选进行快速筛选：Native Batch、BF16、局部编译，以及静态周期图分桶。

## 最终结论

```text
NATIVE_BATCH_GO=False
BF16_GO=False
PARTIAL_COMPILE_GO=False
STATIC_PERIODIC_GRAPH_MVP_RECOMMENDED=True
```

## 主要结果

- Native B4/B8 吞吐约为 B1 的 `3× / 4.2×`，但 composition、目标命中和 RMSD 不等价，因此不能作为零质量风险加速。
- BF16 未通过严格正确性与稳定性要求。
- 局部 `torch.compile` 没有形成可用端到端收益。
- 基于 profiling 的静态周期图分桶估计约 `1.0985×`，因此进入独立 MVP 实测。

## 工作过程

1. 冻结 B1 FP32 完整 Predictor/Corrector 为数值基线。
2. 对 batch、精度、编译与图构建分别做微基准。
3. 对可能改变输出的路线加入逐 seed 科学一致性检查。
4. 淘汰质量不等价的高吞吐路线。
5. 将唯一低风险候选 Static Periodic Graph 交给单独分支验证。

## 如何理解

Native Batch 的高吞吐是真实的，但它改变了随机数消费和生成分布；本项目的第二创新目标要求尽量保持质量，因此不能只以 samples/hour 判定。

## 证据与后续

- [Fast-Gate 最终报告](research/spg_fastgate/artifacts/reports/final_report.md)
- 实测 MVP：`feature/spg-static-periodic-graph-mvp`
- [论文归档分支](https://github.com/du17183/mattergen_v1/tree/archive/thesis-analysis-package-v1)
