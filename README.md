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
