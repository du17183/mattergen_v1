# A0 + E3-G 第二次独立验证

> 当前分支：`feature/a0-e3g-independent64`

本分支使用与模型训练、筛选和第一次兼容性测试均无重叠的新 seeds，复核 A0 + E3-G 的组合效果。

## 冻结结论

```text
A0_E3G_INDEPENDENT64_GO=True
FROZEN_COMMIT=22e1db74a59476562f1f746cd4210b9420cbdf05
SEEDS=50000–50063
N=64
```

| 指标 | A0 + E3-G 相对 A0 |
|---|---:|
| 预松弛最大力 | **-19.02%** |
| Relaxation RMSD | -1.66% |
| 最大力 bootstrap 95% CI | [-0.10221, -0.01070] |
| Wilcoxon p | 0.000587 |
| Win / Tie / Loss | 35 / 18 / 11 |

## 工作过程

1. 在运行前冻结 seeds、模型、门控器、阈值和评估脚本。
2. 核对新 seeds 与训练、开发和历史正式范围交集为零。
3. 运行严格配对的 A0 与 A0 + E3-G。
4. 完成独立 MatterSim relaxation。
5. 计算 bootstrap、Wilcoxon 和逐样本 win/tie/loss。

## 结论用途

该分支是对第一次 64-seed 兼容性实验的独立复现。两次实验的最大力降幅分别为 27.10% 和 19.02%，方向一致；因此它强化了组合模块的可复现性，但仍不能替代 DFT 验证。

## 科学边界

```text
STABILITY_SOURCE=MatterSim-5M surrogate
DFT_VERIFIED=False
PROPERTY_TARGET_VERIFIED=False
```

完整项目脉络见[论文归档分支](https://github.com/du17183/mattergen_v1/tree/archive/thesis-analysis-package-v1)。

## 独立性设计

本次不是从第一次 64-seed 中挑选有利样本，而是在运行前冻结 `50000–50063`：

```text
train/development seeds ∩ 50000–50063 = ∅
formal Q3 seeds ∩ 50000–50063 = ∅
first compatibility seeds ∩ 50000–50063 = ∅
```

A0 和 A0+E3-G 使用相同 seed、初始状态、采样配置和 GPU 映射，差异只来自生成完成后的 E3-G 模块。

## 实现位置

| 文件 | 内容 |
|---|---|
| [`research/a0_e3g_independent64.py`](research/a0_e3g_independent64.py) | 独立 seed 契约、配对 pipeline、统计与 final gate |
| [`research/a0_e3g_compat64.py`](research/a0_e3g_compat64.py) | 复用已验证的组合运行基础设施 |
| [`research/postgen_fastgate/refiner_eval.py`](research/postgen_fastgate/refiner_eval.py) | E3-G 修正 |
| [`tests/test_a0_e3g_independent64.py`](tests/test_a0_e3g_independent64.py) | 独立性、恢复、质量和统计测试 |

## 数据索引

- [Seed 审计](reports/a0_e3g_independent64/seed_audit.json)
- [冻结 manifest](reports/a0_e3g_independent64/frozen_manifest.md)
- [A0 逐结构指标](reports/a0_e3g_independent64/A0/official_metrics_per_structure.csv)
- [组合逐结构指标](reports/a0_e3g_independent64/A0_E3G/official_metrics_per_structure.csv)
- [配对统计](reports/a0_e3g_independent64/paired_statistics.csv)
- [统计解释](reports/a0_e3g_independent64/statistics_report.md)
- [最终报告](reports/a0_e3g_independent64/final_report.md)

## 复现入口

```bash
python -m pytest tests/test_a0_e3g_independent64.py -q
python -m research.a0_e3g_independent64 status
```

完整 pipeline 需要服务器权重、MatterSim 环境和原始结构缓存；GitHub 中的数据足以只读复核逐 seed 效果与统计。
