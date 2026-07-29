# GemNet 融合与持久化 Worker 加速 Fast-Gate

> 当前分支：`feature/gemnet-fused-inference-fastgate`

本分支快速验证两类 GPU 推理加速：GemNet 局部编译/融合，以及不改变模型数值的持久化多 Worker 调度。

## 最终结论

```text
GPU_ACCELERATION_NO_GO=True
QUALITY_CHANGE_EXPECTED=False
```

## Route 1：局部编译

| 指标 | 结果 | 门槛 |
|---|---:|---:|
| Kernel chain | 1.124× | ≥1.25× |
| 完整 forward | 1.009× | ≥1.08× |
| 数值等价 | 未通过 | 必须通过 |

因此没有进入 8-seed 生成。

## Route 2：持久化 Worker

| Workers/GPU | 8 GPU 总吞吐 | 相对 1 Worker | 中位延迟 | 位级一致 |
|---:|---:|---:|---:|---:|
| 1 | 452.973 samples/h | 1.000× | 60.34 s | 是 |
| 2 | 526.820 samples/h | 1.163× | 104.42 s | 是 |
| 4 | 535.090 samples/h | 1.181× | 195.30 s | 是 |

2 workers/GPU 是较合理的工程配置，但总加速未达到冻结的 1.25× 论文门槛。

## 工作过程

1. 分析 GemNet forward 热点和动态图瓶颈。
2. 隔离 K2 kernel chain 进行局部编译微基准。
3. 检查数值等价和完整 forward 收益。
4. 在 8 GPU 上测试 1/2/4 持久化 workers。
5. 以 wall-clock 吞吐、延迟和逐 seed hash 判定。

## 证据

- [最终报告](research/gemnet_fused_fastgate/results_summary/final_report.md)
- [论文归档分支](https://github.com/du17183/mattergen_v1/tree/archive/thesis-analysis-package-v1)
