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

## 实现方式

MPS 测试不改 MatterGen 代码路径。`mps_control.sh` 为当前用户建立独立 pipe/log 目录并安全启停 MPS server；`runtime.py` 启动两个持久化 B1 worker，并通过 `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50` 分配 active threads。

S0 与 S1 使用相同 16 seeds，每种配置预热后重复 3 轮。科学输出只记录一次；性能以总 wall-clock 计算，不依赖 MPS 下单进程 GPU 利用率归因。

## 代码与数据

| 文件 | 内容 |
|---|---|
| [`mps_control.sh`](research/mps_fastgate/mps_control.sh) | 用户态 MPS server 启停和归属检查 |
| [`runtime.py`](research/mps_fastgate/runtime.py) | OFF/ON worker 运行时 |
| [`benchmark.py`](research/mps_fastgate/benchmark.py) | 重复计时、吞吐和延迟统计 |
| [`tests/test_mps_fastgate.py`](tests/test_mps_fastgate.py) | 配置、清理和判定规则测试 |
| [`single_gpu_results.csv`](research/mps_fastgate/artifacts/single_gpu_results.csv) | 三轮原始性能结果 |
| [`bitwise_audit.json`](research/mps_fastgate/artifacts/bitwise_audit.json) | 48/48 逐 seed 位级检查 |

## 复现入口

```bash
bash research/mps_fastgate/scripts/status.sh
python -m pytest tests/test_mps_fastgate.py -q
```

完整 runner 为 `research/mps_fastgate/scripts/run.sh`。它会检查其他用户 MPS 服务并在退出时清理本项目 server；不能使用 sudo、GPU reset 或干扰其他进程。
