# FN-PRA Phase 1

> 当前分支：`feature/fn-pra`

本分支验证静态 FN-PRA：在 `dft_mag_density` 条件 MatterGen + Adaptive CFG 上，用 CHGNet Teacher 对最后一个 GemNet block 做 atom-level representation alignment。

## 最终结论

```text
P1_STATIC_REPA_NO_GO=True
PHASE1_ENGINEERING_GO=False
PHASE1_SCIENTIFIC_GO=False
TRAINING_STEPS=5000
N=32
```

| 指标 | P1 相对 A0 |
|---|---:|
| Composition validity | -6.25 pp |
| Structure validity | 不变 |
| Stable | -6.25 pp |
| 平均 E-hull | +0.003786 eV/atom |
| NUS | +6.25 pp |
| Novel | +21.875 pp |
| Relaxation RMSD | **-28.68%** |
| 中位生成耗时 | -1.769% |

## 工作过程

1. 验证 CHGNet Teacher 在线特征和缓存映射。
2. 实现最后层静态 atom-level REPA 与 EA-NCE。
3. 只训练约 91k 参数，完成 5,000-step 微调。
4. 进行确定性、无 Teacher 推理和 checkpoint 恢复测试。
5. 完成 32-seed A0/P1 配对评估。

## 如何理解

P1 明显降低 RMSD 并提高 NUS/Novel，但 Composition 和 Stable 各下降 6.25 pp，未通过质量安全门槛。该结果推动了后续无条件 CrystalREPA 隔离复现，而不是继续堆叠更复杂模块。

## 证据

- [Phase 1 最终报告](research/fn_pra/reports/phase1/phase1_final_report.md)
- [论文归档分支](https://github.com/du17183/mattergen_v1/tree/archive/thesis-analysis-package-v1)

## 方法实现

P1 在条件 MatterGen 的最后一个 GemNet block 后接静态 atom-level 对齐模块，以 CHGNet Teacher cache 监督 Student 表征。主干大部分冻结，只训练约 91k 参数；采样仍使用 A0 Adaptive CFG。

```text
MP-20 structure mapping
→ CHGNet Teacher feature cache
→ noisy MatterGen Student feature
→ EA-NCE alignment loss + diffusion loss
→ 5,000-step fine-tune
→ 导出无 Teacher 推理模型
```

## 实现位置

| 文件 | 内容 |
|---|---|
| [`phase1_common.py`](research/fn_pra/phase1_common.py) | 路径、冻结配置和共同数据结构 |
| [`teacher_cache_worker.py`](research/fn_pra/teacher_cache_worker.py) | CHGNet atom feature cache |
| [`train_v1.py`](research/fn_pra/train_v1.py) | P1 训练入口 |
| [`validate_v1_integration.py`](research/fn_pra/validate_v1_integration.py) | 映射、推理和 loss 验证 |
| [`run_thirty_two_seed_generation.py`](research/fn_pra/run_thirty_two_seed_generation.py) | 32-seed A0/P1 配对生成 |
| [`mattergen/tests/test_fn_pra.py`](mattergen/tests/test_fn_pra.py) | 对齐、mask、feature disable 和 checkpoint 测试 |

## 数据索引

- [数据映射审计](research/fn_pra/reports/phase1/data_audit.md)
- [Teacher 在线验证](research/fn_pra/reports/phase1/online_teacher_validation.md)
- [训练曲线](research/fn_pra/reports/phase1/training_curves.csv)
- [A0/P1 质量对比](research/fn_pra/reports/phase1/quality_comparison.csv)
- [配对统计](research/fn_pra/reports/phase1/paired_statistics.csv)
- [最终报告](research/fn_pra/reports/phase1/phase1_final_report.md)

## 运行与测试

```bash
bash research/fn_pra/reports/phase1/status_phase1.sh
python -m pytest mattergen/tests/test_fn_pra.py -q
```

完整训练依赖未上传的 Teacher cache 与 checkpoint。README 中的正向 RMSD、NUS/Novel 和负向 Composition/Stable 必须一起报告，不能只摘录有利指标。
