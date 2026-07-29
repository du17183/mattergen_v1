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

## 复现配置

R1 使用无条件 MP-20 checkpoint，关闭 CFG/Adaptive CFG，将 Student 的中间 GemNet block 与 CHGNet atom-level Teacher 表征对齐。EA-NCE 中同结构同 atom index 为正样本；同元素非对角 atom 从负样本中排除。

训练分两段：1,000-step smoke 检查 loss、cosine 和 diffusion validation，再在曲线仍改善时继续到上限 10,000 steps。推理导出不包含 Teacher。

## 实现位置

| 文件 | 内容 |
|---|---|
| [`configuration.py`](research/crystalrepa_repro/configuration.py) | block、loss、训练与路径配置 |
| [`train_repro.py`](research/crystalrepa_repro/train_repro.py) | DDP 训练、checkpoint 和验证 |
| [`validate_ddp_ea_nce.py`](research/crystalrepa_repro/validate_ddp_ea_nce.py) | EA-NCE mask 与 all-gather 验证 |
| [`export_inference.py`](research/crystalrepa_repro/export_inference.py) | 去除 Teacher 的推理 checkpoint |
| [`run_evaluation_pipeline.py`](research/crystalrepa_repro/run_evaluation_pipeline.py) | U0/R1 配对生成、relax 与统计 |
| [`mattergen/tests/test_crystalrepa.py`](mattergen/tests/test_crystalrepa.py) | feature disable、一致性、映射和 checkpoint 测试 |

## 数据索引

- [论文配置核验](research/crystalrepa_repro/artifacts/reports/frozen/paper_config_verified.md)
- [Cache 复用审计](research/crystalrepa_repro/artifacts/reports/cache_reuse_audit.json)
- [10k 训练摘要](research/crystalrepa_repro/artifacts/reports/training_summary_10000.json)
- [U0 逐结构指标](research/crystalrepa_repro/artifacts/reports/U0/official_metrics_per_structure.csv)
- [R1 逐结构指标](research/crystalrepa_repro/artifacts/reports/R1/official_metrics_per_structure.csv)
- [配对统计](research/crystalrepa_repro/artifacts/reports/paired_statistics.csv)

## 运行与测试

```bash
bash research/crystalrepa_repro/ops/status_repro.sh
python -m pytest mattergen/tests/test_crystalrepa.py -q
```

`run_repro.sh` 需要未上传的无条件 checkpoint、Teacher cache 和 MatterSim 权重。GitHub 中保留配置、训练摘要和 64-seed 统计，但不提交训练 checkpoint。
