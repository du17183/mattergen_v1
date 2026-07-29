# NVIDIA MPS 推理加速 Fast-Gate

> 当前分支：`feature/mps-runtime-fastgate`

本分支只回答一个问题：MPS 能否在保持 MatterGen 单轨迹位级一致的情况下，为现有 2 workers/GPU 运行时带来额外吞吐。

## 最终结论

```text
FINAL_STATE=MPS_NO_GO
MPS_BITWISE_EQUIVALENT=True
EIGHT_GPU_STARTED=False
```

| 配置 | 单 GPU 吞吐 |
|---|---:|
| S0：MPS OFF，2 workers | 71.0898 samples/h |
| S1：MPS ON，2 workers，50% active threads | 70.7855 samples/h |
| 增量 | **-0.428%** |

16 个 seeds、3 次正式重复，共 48/48 逐 seed 位级一致，但吞吐没有正增量。按预设规则，未测试 4 workers 或 8 GPU。

## 工作过程

1. 审计当前用户权限、MPS server 和 GPU 可用性。
2. 完成 MPS server 启停和 MatterGen 最小连接测试。
3. 固定 GPU 0、FP32、完整 Predictor/Corrector 和相同 seeds。
4. 对 MPS OFF/ON 各预热并重复三轮。
5. 比较 random tape、结构 hash、吞吐和任务延迟。
6. 正常关闭本项目 MPS server，确认 GPU worker 为零。

## 如何理解

MPS 不改变科学输出，但也没有释放额外吞吐，因此只能作为已排除的部署尝试，不能作为论文加速创新。

## 证据

- [最终报告](research/mps_fastgate/artifacts/final_report.md)
- [论文归档分支](https://github.com/du17183/mattergen_v1/tree/archive/thesis-analysis-package-v1)
