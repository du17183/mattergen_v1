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
