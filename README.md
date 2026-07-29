# A0 + E3-G 兼容性验证

> 当前分支：`feature/a0-e3g-compatibility64`

本分支验证两个创新点能否串行组合：

```text
A0 Adaptive CFG
→ E3-G Learned-Gated Post-generation Refinement
```

## 冻结结论

```text
A0_E3G_COMPATIBILITY_GO=True
FROZEN_COMMIT=ba2303c284210fdae0a35bb0153a8ef3af45a54c
SEEDS=41000–41063
N=64
```

| 指标 | A0 + E3-G 相对 A0 |
|---|---:|
| 预松弛最大力 | **-27.10%** |
| Relaxation RMSD | -1.93% |
| E-hull | 基本不变 |
| Stable / NUS | 不变 |

## 工作过程

1. 固定 A0 的正式 Adaptive CFG 参数。
2. 固定 E3-G 的后处理配置，不重新训练 MatterGen 主干。
3. 对 64 个新 seeds 生成严格配对的 A0 与 A0 + E3-G 结构。
4. 检查 initial state、原子序列和评估配置。
5. 完成 MatterSim relaxation、配对统计和质量安全门槛判定。

## 结论用途

该实验说明两个创新点在算法接口上兼容，E3-G 能在不明显损害 A0 质量指标的情况下进一步降低预松弛最大力。它是组合兼容性证据，不替代两个创新点各自的 256-seed 正式验证。

第二次独立 64-seed 复现位于 `feature/a0-e3g-independent64`。

## 科学边界

```text
STABILITY_SOURCE=MatterSim-5M surrogate
DFT_VERIFIED=False
PROPERTY_TARGET_VERIFIED=False
```

完整项目脉络见[论文归档分支](https://github.com/du17183/mattergen_v1/tree/archive/thesis-analysis-package-v1)。
