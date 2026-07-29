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

## 验证路线

所有候选先在 B1 FP32 完整 Predictor/Corrector 上建立性能和输出基线：

1. Native Batch：B4/B8 一次处理多条轨迹。
2. BF16：模型与关键张量降低精度。
3. Partial compile：只编译可隔离子模块。
4. Static Periodic Graph：按图形状分桶并复用邻接。

性能提升只有在 generation success、composition、目标命中、结构与 relaxation 指标共同通过时才算 GO。

## 实现位置

| 文件 | 内容 |
|---|---|
| [`run_performance_baseline.py`](research/spg_fastgate/run_performance_baseline.py) | B1/B4/B8 性能基线 |
| [`run_quality_generation.py`](research/spg_fastgate/run_quality_generation.py) | 质量配对生成 |
| [`bf16_probe.py`](research/spg_fastgate/bf16_probe.py) | BF16 正确性与速度 probe |
| [`compile_audit.py`](research/spg_fastgate/compile_audit.py) | compile 可行性审计 |
| [`profiler_probe.py`](research/spg_fastgate/profiler_probe.py) | 热点占比与静态图收益估计 |
| [`finalize_fastgate.py`](research/spg_fastgate/finalize_fastgate.py) | 统一 gate 判定 |

## 数据索引

- [性能基线](research/spg_fastgate/artifacts/reports/performance_baseline.md)
- [Profiler 分解](research/spg_fastgate/artifacts/reports/profiler_breakdown.md)
- [Native B4 质量报告](research/spg_fastgate/artifacts/reports/b4_quality_report.md)
- [BF16 报告](research/spg_fastgate/artifacts/reports/bf16_report.md)
- [Compile 报告](research/spg_fastgate/artifacts/reports/compile_report.md)
- [最终 Fast-Gate 报告](research/spg_fastgate/artifacts/reports/final_report.md)

## 复现入口

```bash
bash research/spg_fastgate/scripts/status_fastgate.sh
```

完整 runner 为 `research/spg_fastgate/scripts/run_fastgate.sh`。Native Batch 的吞吐提升不能被表述为“质量不变”，因为逐 seed 科学输出没有通过等价门槛。
