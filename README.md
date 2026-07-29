# Static Periodic Graph MVP

> 当前分支：`feature/spg-static-periodic-graph-mvp`

本分支实测“静态周期图分桶”能否复用周期邻接结构，减少 MatterGen 推理中的动态图构建成本，同时保持严格数值一致。

## 最终结论

```text
STATIC_GRAPH_CORRECTNESS_GO=True
SINGLE_BUCKET_PERFORMANCE_GO=False
SINGLE_BUCKET_NO_GO=True
EIGHT_SEED_STARTED=False
```

## 主要结果

| 指标 | 结果 |
|---|---:|
| 严格图构建等价测试 | 10,000/10,000 通过 |
| 联合 CFG 位级一致 | 64/64 通过 |
| Static graph builder | 1.756× |
| Builder 门槛 | 2.25× |
| 完整 forward | 0.963526× |
| 端到端估计 | 0.9874× |

静态 builder 本身更快，但缓存、布局转换和下游执行抵消了收益，完整 forward 约慢 3.79%。按规则停止，没有运行 8-seed。

## 工作过程

1. 从 profiling 结果定位周期图构建热点。
2. 实现单 bucket 的静态邻接和安全 fallback。
3. 使用 10,000 个图样本验证边、offset 和周期映射。
4. 检查联合 CFG 的逐元素位级一致性。
5. 分别计时 builder、完整 GemNet forward 和端到端占比。

## 如何理解

该分支证明了图等价可以做到，但也证明在当前小图、batch=1、Blackwell GPU 上，图构建并不是足以支撑论文加速比的主瓶颈。

## 证据

- [最终报告](research/spg_static_mvp/final_report.md)
- [论文归档分支](https://github.com/du17183/mattergen_v1/tree/archive/thesis-analysis-package-v1)

## 实现原理

标准路径每次 forward 根据当前 cell、position 和 cutoff 动态构建周期邻接。MVP 先记录图形状分布，选择单一 bucket，预分配 edge/offset 容量，并用 reference builder 做严格语义校验：

```text
record graph shapes
→ freeze one safe-capacity bucket
→ static edge/offset buffers
→ overflow or mismatch uses reference fallback
→ compare edges, offsets, distances and final score
```

## 实现位置

| 文件 | 内容 |
|---|---|
| [`static_builder.py`](research/spg_static_mvp/static_builder.py) | 静态 buffer、bucket 和 fallback |
| [`reference_graph.py`](research/spg_static_mvp/reference_graph.py) | 原始图语义参考 |
| [`capacity_analysis.py`](research/spg_static_mvp/capacity_analysis.py) | bucket 容量分析 |
| [`numerical_equivalence.py`](research/spg_static_mvp/numerical_equivalence.py) | 10k 图和 CFG 数值对齐 |
| [`builder_benchmark.py`](research/spg_static_mvp/builder_benchmark.py) | builder 微基准 |
| [`forward_benchmark.py`](research/spg_static_mvp/forward_benchmark.py) | 完整 GemNet forward 基准 |
| [`test_static_builder.py`](research/spg_static_mvp/tests/test_static_builder.py) | 边、offset、overflow 和 fallback 测试 |

## 数据索引

- [原始图语义](research/spg_static_mvp/artifacts/correctness/original_graph_semantics.md)
- [形状与 bucket](research/spg_static_mvp/artifacts/correctness/shape_distribution.md)
- [10k 等价报告](research/spg_static_mvp/artifacts/correctness/equivalence_report.md)
- [Builder 优化基准](research/spg_static_mvp/artifacts/performance/builder_benchmark_optimized.md)
- [完整 forward 基准](research/spg_static_mvp/artifacts/performance/forward_benchmark.md)
- [最终判定](research/spg_static_mvp/artifacts/final/final_report.md)

## 复现入口

```bash
bash research/spg_static_mvp/scripts/status_mvp.sh
python -m pytest research/spg_static_mvp/tests/test_static_builder.py -q
```

完整 runner 为 `research/spg_static_mvp/scripts/run_mvp.sh`。严格正确性通过，但 builder 收益不足且完整 forward 变慢，因此没有进入生成质量实验。
